import logging

import numpy as np

from src.model import predict_stint

logger = logging.getLogger(__name__)


def compute_pit_window(
    projected_times: np.ndarray,
    tyre_life_range: np.ndarray,
    pit_loss: float,
    model_rmse: float = 0.7,
) -> dict:
    """
    Calcula la ventana optima de pit stop.

    Logica:
        delta[i]     = projected_times[i] - projected_times[0]
                       (tiempo extra pagado en la vuelta i respecto a neumatico nuevo)
        delta_cum[i] = suma acumulada de delta[0..i]
                       (coste total acumulado por seguir en pista)

    Cuando delta_cum >= pit_loss el coste ya iguala o supera la perdida del pit
    stop, lo que indica que hubiera sido mas rapido haber parado antes.

    Devuelve:
        optimal_lap      : TyreLife en el punto de cruce
        window_start     : optimal_lap - 2
        window_end       : optimal_lap + 2
        delta_cumulative : array de coste acumulado
        confidence_laps  : incertidumbre +- en vueltas basada en RMSE del modelo
        triggered        : True si se encontro un cruce, False si la degradacion
                           nunca supera el umbral
    """
    ref = projected_times[0]
    delta = projected_times - ref          # coste por vuelta vs neumatico nuevo
    delta_cum = np.cumsum(delta)            # coste acumulado

    crossing = np.where(delta_cum >= pit_loss)[0]
    triggered = len(crossing) > 0

    if triggered:
        optimal_idx = int(crossing[0])
    else:
        optimal_idx = len(tyre_life_range) - 1
        logger.warning(
            "El coste acumulado nunca supera pit_loss=%.1f s "
            "(max acumulado: %.2f s). Ventana basada en ultimo TyreLife disponible.",
            pit_loss, float(delta_cum[-1]),
        )

    optimal_lap = int(tyre_life_range[optimal_idx])

    # Confianza: +-laps = RMSE / tasa-de-degradacion-en-el-cruce
    if triggered and optimal_idx > 0:
        avg_deg_rate = float(delta_cum[optimal_idx]) / optimal_idx  # s/vuelta
        confidence_laps = max(1, round(model_rmse / max(avg_deg_rate, 0.01)))
    else:
        confidence_laps = 5

    result = {
        "optimal_lap": optimal_lap,
        "window_start": max(int(tyre_life_range[0]), optimal_lap - 2),
        "window_end": min(int(tyre_life_range[-1]), optimal_lap + 2),
        "delta_cumulative": delta_cum,
        "confidence_laps": confidence_laps,
        "triggered": triggered,
    }

    logger.info(
        "Ventana optima: TL=%d  [%d – %d]  (confianza +-%d vueltas)  triggered=%s",
        optimal_lap,
        result["window_start"],
        result["window_end"],
        confidence_laps,
        triggered,
    )
    return result


def compute_undercut_window(
    my_projected_times: np.ndarray,
    rival_stop_tyre_life: int,
    pit_loss: float,
    tyre_life_range: np.ndarray,
) -> dict:
    """
    Si el rival para cuando mi neumatico tiene rival_stop_tyre_life vueltas,
    calcula cuantas vueltas mas puedo aguantar antes de perder la posicion.

    Logica:
        - El rival sale de boxes con neumatico nuevo y absorbe el coste pit_loss.
        - Cada vuelta que yo sigo, el rival recupera la diferencia entre su tiempo
          (neumatico nuevo ~ projected_times[0]) y el mio (degradado).
        - Cuando la ganancia acumulada del rival >= pit_loss, me ha adelantado en
          tiempo: en ese momento ya deberia haber parado.

    Parametros:
        rival_stop_tyre_life : mi TyreLife cuando el rival entra a boxes
    """
    # Indice donde mi TyreLife == rival_stop_tyre_life
    start_idx = int(np.searchsorted(tyre_life_range, rival_stop_tyre_life))
    start_idx = min(start_idx, len(tyre_life_range) - 1)

    fresh_ref = my_projected_times[0]

    # Diferencia de tiempo por vuelta: rival (fresco) vs yo (desgastado)
    # positivo = rival gana tiempo sobre mi
    gain_per_lap = my_projected_times[start_idx:] - fresh_ref
    rival_gap_cum = np.cumsum(gain_per_lap)

    # Crossing: rival recupera los pit_loss que perdio entrando a boxes
    crossing = np.where(rival_gap_cum >= pit_loss)[0]

    if len(crossing) == 0:
        laps_remaining = len(gain_per_lap)
        end_idx = len(tyre_life_range) - 1
        logger.info(
            "Undercut: el rival no logra adelantarme en los %d TyreLife disponibles.",
            len(gain_per_lap),
        )
    else:
        laps_remaining = int(crossing[0])
        end_idx = min(start_idx + laps_remaining, len(tyre_life_range) - 1)
        logger.info(
            "Undercut: rival adelanta en %d vueltas tras su pit (TL=%d -> TL=%d).",
            laps_remaining,
            int(tyre_life_range[start_idx]),
            int(tyre_life_range[end_idx]),
        )

    return {
        "laps_remaining": laps_remaining,
        "must_pit_tyre_life": int(tyre_life_range[end_idx]),
        "rival_gap_cumulative": rival_gap_cum,
        "rival_stop_tyre_life": int(tyre_life_range[start_idx]),
    }


def sensitivity_analysis(
    model,
    scaler,
    tyre_life_range: np.ndarray,
    base_conditions: dict,
    temp_deltas: list[float],
    pit_loss: float,
) -> dict:
    """
    Analiza como cambia la ventana optima si la pista esta X grados mas caliente/fria.

    base_conditions debe contener:
        temp_forecast_fn : callable(session_time_s) -> TrackTemp
        lap_start_times  : np.ndarray de tiempos de inicio (segundos)
        avg_air          : float
        avg_humidity     : float
        avg_wind         : float
        session_baseline : float  (opcional, default 0)

    Para Abu Dhabi: temp_deltas = [-5, 0, +5, +10]

    Devuelve dict { delta_t: { "optimal_lap", "projected_times", "delta_cumulative" } }
    """
    base_fn = base_conditions["temp_forecast_fn"]
    results = {}

    logger.info("=== Analisis de sensibilidad de temperatura ===")
    for delta_t in temp_deltas:
        shifted_fn = lambda t, d=delta_t: base_fn(t) + d

        proj = predict_stint(
            model,
            scaler,
            tyre_life_range,
            shifted_fn,
            base_conditions["lap_start_times"],
            base_conditions["avg_air"],
            base_conditions["avg_humidity"],
            base_conditions["avg_wind"],
            session_baseline=base_conditions.get("session_baseline", 0.0),
        )

        pit_window = compute_pit_window(proj, tyre_life_range, pit_loss)

        results[delta_t] = {
            "optimal_lap": pit_window["optimal_lap"],
            "projected_times": proj,
            "delta_cumulative": pit_window["delta_cumulative"],
            "triggered": pit_window["triggered"],
        }

        logger.info(
            "  dT=%+.0f C  -> ventana TL=%d  [%d-%d]  triggered=%s",
            delta_t,
            pit_window["optimal_lap"],
            pit_window["window_start"],
            pit_window["window_end"],
            pit_window["triggered"],
        )

    return results
