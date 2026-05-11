#!/usr/bin/env python3
"""
Modelo de degradacion de neumaticos F1 — calculo de ventana optima de pit stop.

Uso:
    python main.py --gp "Abu Dhabi" --year 2021 --driver VER
    python main.py --gp "Abu Dhabi" --year 2021 --driver VER --compound HARD
    python main.py --gp "Monaco"    --year 2023 --driver ALO --no-show
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    TRAINING_YEARS,
    get_clean_laps,
    load_sessions,
    measure_pit_loss,
)
from src.model import FEATURE_NAMES, cross_validate_by_year, predict_stint, train_model
from src.plots import (
    backtest_vs_reality,
    plot_degradation,
    plot_delta_cumulative,
    plot_feature_importance,
    plot_sensitivity,
    plot_temperature_evolution,
)
from src.strategy import compute_pit_window, sensitivity_analysis
from src.weather import build_temp_forecast, enrich_laps_with_weather

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modelo de degradacion de neumaticos F1"
    )
    p.add_argument("--gp",       default="Abu Dhabi",
                   help="Nombre del GP  (default: Abu Dhabi)")
    p.add_argument("--year",     type=int, default=2021,
                   help="Anio objetivo  (default: 2021)")
    p.add_argument("--driver",   default="VER",
                   help="Codigo del piloto  (default: VER)")
    p.add_argument("--compound", default=None,
                   choices=["SOFT", "MEDIUM", "HARD"],
                   help="Compuesto a analizar. Si se omite, analiza todos los disponibles.")
    p.add_argument("--no-show",  action="store_true",
                   help="No mostrar graficos interactivos (solo guardar PNG)")
    p.add_argument("--out-dir",  default="outputs",
                   help="Directorio de salida  (default: outputs/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_compounds(session, driver: str) -> list[str]:
    """
    Devuelve los compuestos secos (SOFT/MEDIUM/HARD) que uso el piloto
    en la sesion, excluyendo INTERMEDIATE y WET.
    """
    laps = session.laps.pick_drivers(driver)
    used = laps["Compound"].str.upper().unique().tolist()
    return [c for c in DRY_COMPOUNDS if c in used]


def _build_training_df(sessions: dict, driver: str, compound: str) -> pd.DataFrame:
    """
    Para cada sesion: limpia vueltas y enriquece con weather.
    Concatena todo en un DataFrame de entrenamiento con columna Year.
    """
    frames = []
    for year, session in sorted(sessions.items()):
        laps = get_clean_laps(session, driver, compound)
        if laps.empty:
            continue
        enriched = enrich_laps_with_weather(laps, session.weather_data)
        if enriched.empty:
            continue
        enriched["Year"] = year
        frames.append(enriched)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stint_params(laps_df: pd.DataFrame, n_extra: int = 5) -> dict:
    """
    Extrae parametros de simulacion de stint desde el DataFrame de vueltas reales:
    - tyre_life_range : 1 .. max_observado + n_extra
    - lap_start_times : tiempos estimados de inicio de cada vuelta (segundos)
    - promedios de AirTemp, Humidity, WindSpeed
    - session_baseline: mediana de LapTimeSec (para des-normalizar predicciones)
    """
    avg_lap = float(laps_df["LapTimeSec"].mean())
    max_tl  = int(laps_df["TyreLife"].max())

    t0_raw = laps_df["LapStartTime"].iloc[0]
    t0 = t0_raw.total_seconds() if hasattr(t0_raw, "total_seconds") else float(t0_raw)

    tl_range = np.arange(1, max_tl + n_extra + 1)
    lap_ts   = t0 + (tl_range - 1) * avg_lap

    return {
        "tyre_life_range":  tl_range,
        "lap_start_times":  lap_ts,
        "avg_air":          float(laps_df["AirTemp"].mean()),
        "avg_humidity":     float(laps_df["Humidity"].mean()),
        "avg_wind":         float(laps_df["WindSpeed"].mean()),
        "session_baseline": float(laps_df["LapTimeSec"].median()),
    }


def _real_pit_tyre_life(session, driver: str, compound: str) -> list[int]:
    """
    TyreLife en el momento del pit stop para ese compuesto.
    Puede ser vacio (si no hubo pit stop, e.g. ultimo stint).
    """
    laps = session.laps.pick_drivers(driver)
    in_laps = laps[laps["PitInTime"].notna() & (laps["Compound"] == compound)]
    return [int(tl) for tl in in_laps["TyreLife"].tolist()]


# ---------------------------------------------------------------------------
# Pipeline por compuesto
# ---------------------------------------------------------------------------

def _run_compound(
    sessions: dict,
    session_target,
    args: argparse.Namespace,
    compound: str,
    logger: logging.Logger,
) -> list:
    """
    Ejecuta el pipeline completo (entrenamiento, prediccion, graficos)
    para un compuesto concreto. Devuelve la lista de figuras generadas.
    """
    import matplotlib.pyplot as plt

    tag     = f"{args.gp.replace(' ', '_')}_{args.year}_{args.driver}_{compound}"
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 62)
    print(f"  F1 Strategy | {args.gp} {args.year} | {args.driver} | {compound}")
    print("=" * 62)

    # --- Dataset de entrenamiento ---
    logger.info("[2/9] Construyendo dataset de entrenamiento (%s)...", compound)
    df_train = _build_training_df(sessions, args.driver, compound)

    if df_train.empty:
        logger.warning(
            "Sin datos validos para %s %s en ninguno de los anios %s. Omitiendo.",
            args.driver, compound, TRAINING_YEARS,
        )
        return []

    logger.info(
        "Dataset listo: %d vueltas, anios %s",
        len(df_train), sorted(df_train["Year"].unique().tolist()),
    )

    # --- Cross-validacion LOYO ---
    logger.info("[3/9] Cross-validacion LOYO...")
    cv_results = cross_validate_by_year(df_train)

    print("\n--- Cross-validacion leave-one-year-out (target normalizado) ---")
    if cv_results:
        for yr, m in sorted(cv_results.items()):
            print(f"  {yr}:  RMSE = {m['rmse']:.3f} s   R2 = {m['r2']:+.4f}   n = {m['n']}")
        mean_rmse = float(np.mean([m["rmse"] for m in cv_results.values()]))
        print(f"  Media RMSE : {mean_rmse:.3f} s")
    else:
        mean_rmse = 0.7
        print("  (< 2 anios de datos - CV no disponible, usando RMSE=0.7 s)")
    print()

    # --- Modelo final ---
    logger.info("[4/9] Entrenando modelo final con todos los anios...")
    model, scaler = train_model(df_train)

    # --- Sesion objetivo ---
    logger.info("[5/9] Preparando sesion objetivo %d...", args.year)
    laps_target = enrich_laps_with_weather(
        get_clean_laps(session_target, args.driver, compound),
        session_target.weather_data,
    )

    if laps_target.empty:
        logger.warning(
            "Sin vueltas limpias de %s %s en %d. Omitiendo.",
            args.driver, compound, args.year,
        )
        return []

    forecast_fn = build_temp_forecast(session_target.weather_data)
    pit_loss    = measure_pit_loss(session_target, args.driver)
    logger.info("pit_loss medido = %.2f s", pit_loss)

    # --- Proyeccion de tiempos ---
    logger.info("[6/9] Proyectando tiempos de vuelta (T dinamica)...")
    sp = _stint_params(laps_target)

    projected = predict_stint(
        model, scaler,
        sp["tyre_life_range"], forecast_fn, sp["lap_start_times"],
        sp["avg_air"], sp["avg_humidity"], sp["avg_wind"],
        session_baseline=sp["session_baseline"],
    )

    # --- Ventana optima ---
    logger.info("[7/9] Calculando ventana optima de pit stop...")
    pit_window = compute_pit_window(
        projected, sp["tyre_life_range"], pit_loss, model_rmse=mean_rmse,
    )

    print(f"--- Ventana optima de pit stop ({compound}) ---")
    print(f"  TL optimo     : {pit_window['optimal_lap']}")
    print(f"  Ventana       : TL {pit_window['window_start']} - {pit_window['window_end']}")
    print(f"  Confianza     : +/-{pit_window['confidence_laps']} vueltas")
    print(f"  Triggered     : {pit_window['triggered']}")
    dc_max = float(pit_window["delta_cumulative"].max())
    print(f"  Delta cum max : {dc_max:.3f} s  (pit_loss = {pit_loss:.1f} s)")
    print()

    # --- Sensibilidad a temperatura ---
    logger.info("[8/9] Analisis de sensibilidad a temperatura...")
    base_conds = {
        "temp_forecast_fn": forecast_fn,
        "lap_start_times":  sp["lap_start_times"],
        "avg_air":          sp["avg_air"],
        "avg_humidity":     sp["avg_humidity"],
        "avg_wind":         sp["avg_wind"],
        "session_baseline": sp["session_baseline"],
    }
    sens_results = sensitivity_analysis(
        model, scaler, sp["tyre_life_range"], base_conds,
        temp_deltas=[-5, 0, +5, +10],
        pit_loss=pit_loss,
    )

    print("--- Sensibilidad a temperatura ---")
    print(f"  {'dT':>5}  {'TL optimo':>10}  {'Triggered':>10}")
    for dt, res in sorted(sens_results.items()):
        print(f"  {dt:+5.0f}C  {res['optimal_lap']:>10}  {str(res['triggered']):>10}")
    print()

    # --- Graficos ---
    logger.info("[9/9] Generando graficos...")
    session_label = f"{args.gp} {args.year}"
    figs = []

    figs.append(plot_degradation(
        laps_target, projected, sp["tyre_life_range"], pit_window,
        compound, args.driver, session_label,
        save_path=str(out_dir / "1_degradacion.png"),
    ))

    figs.append(plot_delta_cumulative(
        pit_window["delta_cumulative"], sp["tyre_life_range"],
        pit_loss, pit_window["optimal_lap"], pit_window["triggered"],
        save_path=str(out_dir / "2_delta_cumulative.png"),
    ))

    pit_times_s = [
        t.total_seconds()
        for t in session_target.laps.pick_drivers(args.driver)["PitInTime"].dropna()
    ]
    figs.append(plot_temperature_evolution(
        session_target.weather_data, pit_times_s,
        save_path=str(out_dir / "3_temperatura.png"),
    ))

    figs.append(plot_sensitivity(
        sens_results, sp["tyre_life_range"], pit_loss, compound,
        save_path=str(out_dir / "4_sensibilidad.png"),
    ))

    figs.append(plot_feature_importance(
        model, scaler,
        save_path=str(out_dir / "5_feature_importance.png"),
    ))

    # --- Backtesting ---
    real_tl = _real_pit_tyre_life(session_target, args.driver, compound)
    backtest_vs_reality(pit_window, real_tl, compound, args.driver, args.year, args.gp)

    print(f"\nGraficos guardados en: {out_dir.resolve()}")
    return figs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    logger = logging.getLogger("main")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Cargar sesiones historicas (una sola vez para todos los compuestos) ---
    logger.info("[1/9] Cargando sesiones historicas (%s)...", TRAINING_YEARS)
    sessions = load_sessions(args.gp, TRAINING_YEARS)

    if args.year not in sessions:
        logger.error("No se pudo cargar el anio objetivo %d. Abortando.", args.year)
        sys.exit(1)

    session_target = sessions[args.year]

    # --- Determinar compuestos a analizar ---
    if args.compound is not None:
        compounds = [args.compound]
    else:
        compounds = _detect_compounds(session_target, args.driver)
        if not compounds:
            logger.error(
                "No se encontraron compuestos secos para %s en %s %d.",
                args.driver, args.gp, args.year,
            )
            sys.exit(1)
        logger.info(
            "Compuestos detectados para %s en %s %d: %s",
            args.driver, args.gp, args.year, compounds,
        )

    # --- Ejecutar pipeline para cada compuesto ---
    all_figs = []
    for compound in compounds:
        figs = _run_compound(sessions, session_target, args, compound, logger)
        all_figs.extend(figs)

    if args.no_show:
        for fig in all_figs:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
