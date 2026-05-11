import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)

WEATHER_COLS = ["TrackTemp", "AirTemp", "Humidity", "WindSpeed"]


def enrich_laps_with_weather(laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas TrackTemp, AirTemp, Humidity, WindSpeed a cada vuelta
    usando join temporal por rango [LapStartTime, Time].

    Para cada vuelta se promedian todas las muestras de weather dentro del rango.
    Fallback: si no hay muestras, se usa la más cercana a LapStartTime.
    Elimina vueltas con Rainfall == True durante cualquier muestra del rango.
    """
    if laps_df.empty:
        return laps_df.copy()

    result = laps_df.copy()

    # Convertir tiempos a segundos flotantes para operaciones vectorizadas
    w_time = weather_df["Time"].dt.total_seconds().values          # (n_weather,)
    l_start = laps_df["LapStartTime"].dt.total_seconds().values    # (n_laps,)
    l_end = laps_df["Time"].dt.total_seconds().values              # (n_laps,)

    # Matriz booleana (n_laps, n_weather): True si la muestra cae dentro de la vuelta
    in_range = (w_time[None, :] >= l_start[:, None]) & (w_time[None, :] <= l_end[:, None])
    n_samples = in_range.sum(axis=1)  # número de muestras por vuelta

    # Índice de la muestra más cercana a LapStartTime (fallback)
    nearest = np.argmin(np.abs(w_time[None, :] - l_start[:, None]), axis=1)

    # --- Variables de temperatura y clima ---
    for col in WEATHER_COLS:
        col_vals = weather_df[col].values.astype(float)
        # Suma de valores en rango para cada vuelta
        range_sum = (in_range * col_vals[None, :]).sum(axis=1)
        means = np.where(
            n_samples > 0,
            range_sum / np.maximum(n_samples, 1),
            col_vals[nearest],
        )
        result[col] = means

    # --- Filtro de lluvia ---
    rainfall_vals = weather_df["Rainfall"].fillna(False).astype(bool).values
    rainy_in_range = (in_range & rainfall_vals[None, :]).any(axis=1)
    rainy_fallback = rainfall_vals[nearest]
    is_rainy = np.where(n_samples > 0, rainy_in_range, rainy_fallback)

    n_rainy = is_rainy.sum()
    if n_rainy > 0:
        logger.info("Eliminando %d vueltas con lluvia.", n_rainy)
    result = result[~is_rainy].reset_index(drop=True)

    # Diagnóstico: vueltas sin muestras exactas (usaron fallback)
    n_fallback = (n_samples == 0).sum()
    if n_fallback > 0:
        logger.debug("%d vueltas usaron muestra de weather más cercana (sin overlap exacto).", n_fallback)

    logger.info(
        "Weather enrichment: %d vueltas → TrackTemp %.1f–%.1f °C",
        len(result),
        result["TrackTemp"].min(),
        result["TrackTemp"].max(),
    )
    return result


def build_temp_forecast(weather_df: pd.DataFrame) -> callable:
    """
    Construye una función T_pred(session_time_seconds) -> float
    usando regresión lineal sobre los datos observados de weather_data.

    En Abu Dabi (nocturna) la pista baja ~12°C de vuelta 1 a vuelta 58.
    Sin esta proyección dinámica el modelo se equivoca hasta 5 vueltas en
    la ventana óptima.

    Usa scipy.stats.linregress sobre weather_df["Time"] vs weather_df["TrackTemp"].
    """
    times = weather_df["Time"].dt.total_seconds().values
    temps = weather_df["TrackTemp"].values.astype(float)

    valid = ~(np.isnan(times) | np.isnan(temps))
    if valid.sum() < 2:
        logger.warning("Datos insuficientes para forecast de temperatura. Devuelve temperatura media.")
        mean_temp = float(np.nanmean(temps))
        return lambda t: mean_temp

    slope, intercept, r_value, _, std_err = linregress(times[valid], temps[valid])

    logger.info(
        "Forecast TrackTemp: %.5f °C/s  (%.2f °C/vuelta@90s), R²=%.3f, σ=%.3f",
        slope, slope * 90, r_value ** 2, std_err,
    )

    def temp_forecast(session_time_seconds: float) -> float:
        """Predice TrackTemp (°C) a partir del tiempo de sesión en segundos."""
        return float(slope * session_time_seconds + intercept)

    # Adjuntar metadatos para inspección
    temp_forecast.slope = slope
    temp_forecast.intercept = intercept
    temp_forecast.r_squared = r_value ** 2

    return temp_forecast


def get_expected_temp_at_lap(lap_start_time, temp_forecast_fn: callable) -> float:
    """
    Evalúa la función de proyección de temperatura en un SessionTime futuro.
    Acepta lap_start_time como timedelta o como float (segundos).
    Usado en la simulación de estrategia para predecir T en vueltas futuras.
    """
    if hasattr(lap_start_time, "total_seconds"):
        secs = lap_start_time.total_seconds()
    else:
        secs = float(lap_start_time)
    return temp_forecast_fn(secs)
