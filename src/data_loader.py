import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[1] / "ff1_cache"
TRAINING_YEARS = [2018, 2019, 2021, 2022, 2023]


def load_sessions(gp_name: str, years: list[int], session_type: str = "R") -> dict:
    """
    Carga varias ediciones del mismo GP.
    Devuelve dict {year: Session} con weather=True cargado.
    Usa ff1_cache/ para no re-descargar.
    """
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    sessions = {}
    for year in years:
        logger.info("Cargando %s %d %s...", gp_name, year, session_type)
        try:
            session = fastf1.get_session(year, gp_name, session_type)
            session.load(weather=True)
            sessions[year] = session
            logger.info("  %d vueltas cargadas", len(session.laps))
        except Exception as exc:
            logger.warning("No se pudo cargar %d %s: %s", year, gp_name, exc)
    return sessions


def get_clean_laps(session, driver: str, compound: str = None) -> pd.DataFrame:
    """
    Filtra las vueltas válidas de un piloto:
    - IsAccurate == True
    - PitInTime es NaT  (no es vuelta de entrada a boxes)
    - PitOutTime es NaT (no es vuelta de salida de boxes)
    - TrackStatus == "1" (pista verde, sin SC ni bandera amarilla)
    - Deleted == False  (si la columna existe)
    - Compound == compound (si se especifica)
    Añade columna LapTimeSec (LapTime en segundos).
    """
    laps = session.laps.pick_drivers(driver).copy()

    mask = laps["IsAccurate"] == True
    mask &= laps["PitInTime"].isna()
    mask &= laps["PitOutTime"].isna()
    mask &= laps["TrackStatus"] == "1"

    if "Deleted" in laps.columns:
        mask &= laps["Deleted"] == False

    if compound is not None:
        mask &= laps["Compound"].str.upper() == compound.upper()

    laps = laps[mask].copy()
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps = laps.dropna(subset=["LapTimeSec", "TyreLife"])

    logger.debug(
        "Piloto %s: %d vueltas limpias%s",
        driver, len(laps), f" ({compound})" if compound else "",
    )
    return laps.reset_index(drop=True)


def measure_pit_loss(session, driver: str) -> float:
    """
    Calcula el pit stop loss neto real desde los datos:
    pit_loss = median(PitOutTime - PitInTime) - median_lap_time

    PitInTime es la hora de sesión en que el coche entra a boxes (al final de la vuelta).
    PitOutTime es la hora de sesión en que el coche sale de boxes (al inicio de la vuelta).
    La diferencia es el tiempo total invertido en el pit stop.
    El pit_loss neto es ese tiempo menos una vuelta limpia de referencia.
    """
    laps = session.laps.pick_drivers(driver)

    in_laps = laps[laps["PitInTime"].notna()].sort_values("LapNumber")
    out_laps = laps[laps["PitOutTime"].notna()].sort_values("LapNumber")

    durations = []
    for _, in_lap in in_laps.iterrows():
        # La vuelta de salida es la primera con LapNumber posterior y PitOutTime definido
        candidates = out_laps[out_laps["LapNumber"] > in_lap["LapNumber"]]
        if candidates.empty:
            continue
        out_lap = candidates.iloc[0]
        delta = (out_lap["PitOutTime"] - in_lap["PitInTime"]).total_seconds()
        # Sanity check: un pit stop en F1 dura entre 15 y 55 segundos en total
        if 15.0 <= delta <= 55.0:
            durations.append(delta)

    if not durations:
        logger.warning(
            "No se encontraron pit stops válidos para %s. Usando 20 s por defecto.", driver
        )
        return 20.0

    # PitOutTime - PitInTime ya representa el tiempo total dentro del pit lane,
    # que es exactamente el pit stop loss neto (comparar con conducir esa sección
    # a velocidad de carrera no es necesario porque las reglas de speed limit lo fijan).
    pit_loss = float(np.median(durations))

    logger.info(
        "Pit stop loss para %s: %.2f s  (n=%d pit stops, rango [%.1f, %.1f] s)",
        driver, pit_loss, len(durations), min(durations), max(durations),
    )
    return pit_loss


def build_training_dataframe(sessions: dict, driver: str, compound: str) -> pd.DataFrame:
    """
    Une las vueltas limpias de todas las ediciones en un solo DataFrame.
    Añade columna 'Year' para la validación cruzada leave-one-year-out.
    """
    frames = []
    for year, session in sorted(sessions.items()):
        df = get_clean_laps(session, driver, compound)
        if df.empty:
            logger.warning("Sin vueltas limpias para %d (%s %s) — año excluido.", year, driver, compound)
            continue
        df["Year"] = year
        frames.append(df)
        logger.info("Año %d: %d vueltas añadidas al dataset.", year, len(df))

    if not frames:
        raise ValueError(
            f"No hay datos válidos para driver='{driver}' compound='{compound}' "
            f"en ninguno de los años cargados."
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Dataset de entrenamiento listo: %d vueltas, años %s",
        len(combined), sorted(combined["Year"].unique().tolist()),
    )
    return combined
