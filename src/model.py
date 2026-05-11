import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "TyreLife",
    "TyreLife_sq",
    "TrackTemp",
    "TyreLife_x_TrackTemp",
    "AirTemp",
    "Humidity",
    "WindSpeed",
]
ALPHAS = [0.1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------

def compute_baselines(df: pd.DataFrame) -> dict:
    """
    Calcula la mediana del tiempo de vuelta por año.
    Se usa para normalizar antes de entrenar (elimina diferencias de ritmo
    entre eras del coche o trazados distintos) y des-normalizar al predecir.
    """
    return df.groupby("Year")["LapTimeSec"].median().to_dict()


def _normalize(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Resta el baseline de cada año a LapTimeSec."""
    df = df.copy()
    df["LapTimeSec"] = df["LapTimeSec"] - df["Year"].map(baselines)
    return df


# ---------------------------------------------------------------------------
# Construcción de features
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Construye la matriz X con las 7 features en el orden:
    [TyreLife, TyreLife^2, TrackTemp, TyreLife*TrackTemp,
     AirTemp, Humidity, WindSpeed]
    """
    tl = df["TyreLife"].values.astype(float)
    tt = df["TrackTemp"].values.astype(float)
    return np.column_stack([
        tl,
        tl ** 2,
        tt,
        tl * tt,
        df["AirTemp"].values.astype(float),
        df["Humidity"].values.astype(float),
        df["WindSpeed"].values.astype(float),
    ])


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame) -> tuple[RidgeCV, StandardScaler]:
    """
    Entrena Ridge regression sobre el DataFrame completo.
    El target se normaliza por año (delta vs mediana del año) para eliminar
    diferencias absolutas de ritmo entre eras del coche o versiones del trazado.
    Devuelve (modelo_entrenado, scaler_ajustado).
    """
    baselines = compute_baselines(df)
    df_norm = _normalize(df, baselines)

    X = build_feature_matrix(df_norm)
    y = df_norm["LapTimeSec"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = RidgeCV(alphas=ALPHAS, cv=5)
    model.fit(X_s, y)

    r2_train = float(model.score(X_s, y))
    logger.info(
        "Modelo entrenado | alpha=%.1f | R2_train=%.4f | n=%d vueltas",
        model.alpha_, r2_train, len(y),
    )
    _log_feature_importance(model, scaler)
    return model, scaler


# ---------------------------------------------------------------------------
# Cross-validación
# ---------------------------------------------------------------------------

def cross_validate_by_year(df: pd.DataFrame) -> dict:
    """
    Leave-one-year-out cross validation sobre tiempos normalizados.

    En cada iteración:
      - El baseline de entrenamiento se calcula solo con los años de train.
      - El baseline de validación se calcula con la mediana del propio año
        excluido (simula conocer el ritmo de la sesión actual, como en la
        aplicación real del modelo).
      - El RMSE y R2 miden la predicción de la DEGRADACION, no del tiempo
        absoluto, lo que hace las métricas útiles y comparables entre años.
    """
    years = sorted(df["Year"].unique())
    results = {}

    logger.info("=== Cross-validacion leave-one-year-out (target normalizado) ===")
    for val_year in years:
        train_df = df[df["Year"] != val_year].copy()
        val_df = df[df["Year"] == val_year].copy()

        if train_df.empty or val_df.empty:
            logger.warning("Anio %d: datos insuficientes, omitido.", val_year)
            continue

        # Baseline de entrenamiento: solo años de train
        train_baselines = compute_baselines(train_df)
        # Baseline de validacion: mediana del propio año excluido
        val_baseline = val_df["LapTimeSec"].median()
        val_df_norm = val_df.copy()
        val_df_norm["LapTimeSec"] = val_df["LapTimeSec"] - val_baseline

        train_df_norm = _normalize(train_df, train_baselines)

        X_train = build_feature_matrix(train_df_norm)
        y_train = train_df_norm["LapTimeSec"].values
        X_val = build_feature_matrix(val_df_norm)
        y_val = val_df_norm["LapTimeSec"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        cv_folds = min(5, len(train_df))
        model = RidgeCV(alphas=ALPHAS, cv=cv_folds)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_val_s)
        rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
        r2 = float(r2_score(y_val, y_pred))

        results[val_year] = {"rmse": rmse, "r2": r2, "n": len(val_df)}
        logger.info(
            "  Excluido %d (n=%2d) | RMSE=%5.3f s | R2=%6.4f | alpha=%.1f",
            val_year, len(val_df), rmse, r2, model.alpha_,
        )

    mean_rmse = np.mean([v["rmse"] for v in results.values()])
    logger.info("RMSE medio LOYO: %.3f s", mean_rmse)
    return results


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def predict_lap_time(
    model: RidgeCV,
    scaler: StandardScaler,
    tyre_life: float,
    track_temp: float,
    air_temp: float,
    humidity: float,
    wind_speed: float,
    session_baseline: float = 0.0,
) -> float:
    """
    Predice el tiempo de vuelta para una combinación de condiciones.
    session_baseline: mediana de tiempos limpios de la sesión actual
                      (añadir para obtener tiempos absolutos).
    """
    tl = float(tyre_life)
    X = np.array([[
        tl,
        tl ** 2,
        float(track_temp),
        tl * float(track_temp),
        float(air_temp),
        float(humidity),
        float(wind_speed),
    ]])
    return float(model.predict(scaler.transform(X))[0]) + session_baseline


def predict_stint(
    model: RidgeCV,
    scaler: StandardScaler,
    tyre_life_range: np.ndarray,
    temp_forecast_fn: callable,
    lap_start_times: np.ndarray,
    avg_air: float,
    avg_humidity: float,
    avg_wind: float,
    session_baseline: float = 0.0,
) -> np.ndarray:
    """
    Predice un array de tiempos de vuelta usando T(vuelta) dinamica.

    tyre_life_range  : vida del neumatico para cada vuelta proyectada
    temp_forecast_fn : callable(session_time_seconds) -> TrackTemp
    lap_start_times  : tiempos de inicio de cada vuelta (segundos o timedelta)
    session_baseline : mediana de tiempos limpios de la sesion actual;
                       permite des-normalizar y obtener tiempos absolutos.
    """
    tl = tyre_life_range.astype(float)

    times_s = np.array([
        t.total_seconds() if hasattr(t, "total_seconds") else float(t)
        for t in lap_start_times
    ])
    tt = np.array([temp_forecast_fn(t) for t in times_s])

    X = np.column_stack([
        tl,
        tl ** 2,
        tt,
        tl * tt,
        np.full_like(tl, avg_air),
        np.full_like(tl, avg_humidity),
        np.full_like(tl, avg_wind),
    ])
    return model.predict(scaler.transform(X)) + session_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_feature_importance(model: RidgeCV, scaler: StandardScaler) -> None:
    """Importancia de cada feature: |coef * std| normalizado."""
    importance = np.abs(model.coef_ * scaler.scale_)
    importance /= importance.sum()
    pairs = sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True)
    logger.info("Importancia de features (normalizada):")
    for name, imp in pairs:
        bar = "#" * int(imp * 30)
        logger.info("  %-24s %s %.3f", name, bar, imp)
