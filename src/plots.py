"""
Capa 5 — visualización y backtesting.
Cada función devuelve el objeto Figure para que el llamador decida si
mostrar (plt.show()) o guardar (fig.savefig(...)).
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.model import FEATURE_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de estilo
# ---------------------------------------------------------------------------

COMPOUND_COLOR = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD": "#ABABAB",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
}

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.edgecolor": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "figure.dpi": 120,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# 1. Degradación principal
# ---------------------------------------------------------------------------

def plot_degradation(
    df_real: pd.DataFrame,
    projected_times: np.ndarray,
    tyre_life_range: np.ndarray,
    pit_window: dict,
    compound: str,
    driver: str,
    session_info: str,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grafico principal de degradacion:
    - Scatter de vueltas reales coloreadas por TrackTemp (cmap RdYlBu_r)
    - Linea del modelo proyectado
    - Banda verde sombreada = ventana optima (si triggered=True)
    - Colorbar de temperatura
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    # --- Scatter: vueltas reales coloreadas por temperatura ---
    sc = ax.scatter(
        df_real["TyreLife"],
        df_real["LapTimeSec"],
        c=df_real["TrackTemp"],
        cmap="RdYlBu_r",
        s=70,
        zorder=5,
        edgecolors="#333333",
        linewidths=0.4,
        label="Vueltas reales",
        vmin=df_real["TrackTemp"].min() - 1,
        vmax=df_real["TrackTemp"].max() + 1,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("TrackTemp (°C)", fontsize=10)

    # --- Linea: tiempos proyectados ---
    color = COMPOUND_COLOR.get(compound.upper(), "#555555")
    ax.plot(
        tyre_life_range,
        projected_times,
        color=color,
        lw=2.5,
        zorder=4,
        label=f"Modelo proyectado ({compound})",
        solid_capstyle="round",
    )

    # --- Ventana optima ---
    if pit_window.get("triggered", False):
        ws = pit_window["window_start"]
        we = pit_window["window_end"]
        ol = pit_window["optimal_lap"]
        ax.axvspan(ws, we, alpha=0.18, color="#00aa44", zorder=2,
                   label=f"Ventana optima TL {ws}–{we}")
        ax.axvline(ol, color="#00cc55", lw=1.8, linestyle="--", zorder=6,
                   label=f"TL optimo = {ol}")
        ax.annotate(
            f"TL={ol}",
            xy=(ol, projected_times[ol - int(tyre_life_range[0])]),
            xytext=(ol + 0.5, projected_times[ol - int(tyre_life_range[0])] + 0.15),
            fontsize=9,
            color="#00cc55",
            arrowprops=dict(arrowstyle="->", color="#00cc55", lw=1),
        )
    else:
        ax.text(
            0.98, 0.05,
            "Sin ventana por degradacion\n(pit stop tactique/SC)",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9, color="#888888",
            style="italic",
        )

    ax.set_xlabel("TyreLife (vueltas en el neumatico)")
    ax.set_ylabel("Tiempo de vuelta (s)")
    ax.set_title(f"Degradacion {compound} — {driver} | {session_info}")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Guardado: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Coste acumulado vs umbral
# ---------------------------------------------------------------------------

def plot_delta_cumulative(
    delta_cum: np.ndarray,
    tyre_life_range: np.ndarray,
    pit_loss: float,
    optimal_lap: int,
    triggered: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Coste acumulado de seguir en pista vs umbral del pit stop.
    Anotacion en el punto de cruce (si triggered=True).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Zona positiva (perdida real) vs negativa (ganancia por combustible)
    ax.fill_between(tyre_life_range, delta_cum, 0,
                    where=(delta_cum >= 0),
                    alpha=0.25, color="#cc3300", label="Coste neto positivo")
    ax.fill_between(tyre_life_range, delta_cum, 0,
                    where=(delta_cum < 0),
                    alpha=0.15, color="#0066cc", label="Ganancia neta (combustible)")

    ax.plot(tyre_life_range, delta_cum, color="#333333", lw=2, zorder=4,
            label="Coste acumulado vs neumatico nuevo")

    # Linea horizontal: umbral pit_loss
    ax.axhline(pit_loss, color="#dd4400", lw=1.8, linestyle="--",
               label=f"Pit stop loss = {pit_loss:.1f} s")
    ax.axhline(0, color="#888888", lw=0.8, linestyle="-")

    # Anotacion en el cruce
    if triggered:
        idx = int(np.searchsorted(tyre_life_range, optimal_lap))
        idx = min(idx, len(tyre_life_range) - 1)
        ax.axvline(optimal_lap, color="#00cc55", lw=1.8, linestyle="--",
                   label=f"Cruce TL={optimal_lap}")
        ax.annotate(
            f"TL={optimal_lap}\n{delta_cum[idx]:.1f} s",
            xy=(optimal_lap, delta_cum[idx]),
            xytext=(optimal_lap + 1, delta_cum[idx] - 1.5),
            fontsize=9, color="#00cc55",
            arrowprops=dict(arrowstyle="->", color="#00cc55", lw=1),
        )
    else:
        ax.text(
            0.5, 0.95,
            f"El coste acumulado max ({delta_cum.max():.1f} s) no supera\n"
            f"el pit loss ({pit_loss:.1f} s) — no hay ventana por degradacion",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9, color="#888888",
            style="italic",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="#cccccc"),
        )

    ax.set_xlabel("TyreLife (vueltas en el neumatico)")
    ax.set_ylabel("Coste acumulado (s)")
    ax.set_title("Coste acumulado de seguir en pista vs pit stop")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Guardado: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Evolucion de temperatura
# ---------------------------------------------------------------------------

def plot_temperature_evolution(
    weather_df: pd.DataFrame,
    pit_session_times: list[float] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    TrackTemp y AirTemp a lo largo de la sesion.
    pit_session_times: lista de tiempos de sesion (segundos) donde se hicieron
                       pit stops (se dibujan lineas verticales).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5))

    t_min = weather_df["Time"].dt.total_seconds() / 60.0

    ax.plot(t_min, weather_df["TrackTemp"],
            color="#E8002D", lw=2, label="TrackTemp (°C)")
    ax.fill_between(t_min, weather_df["TrackTemp"],
                    weather_df["TrackTemp"].min() - 1,
                    alpha=0.08, color="#E8002D")

    ax.plot(t_min, weather_df["AirTemp"],
            color="#3399ff", lw=1.8, linestyle="--", label="AirTemp (°C)")

    # Lineas verticales en pit stops
    if pit_session_times:
        for i, t_s in enumerate(pit_session_times):
            t_m = t_s / 60.0
            ax.axvline(
                t_m, color="#888888", lw=1.2, linestyle=":",
                label="Pit stop" if i == 0 else None,
            )
            ax.annotate(
                f"Pit\n{t_m:.0f} min",
                xy=(t_m, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else weather_df["AirTemp"].min()),
                xytext=(t_m + 0.5, weather_df["AirTemp"].min() + 0.3),
                fontsize=8, color="#666666",
            )

    ax.set_xlabel("Tiempo de sesion (minutos)")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title("Evolucion de temperatura durante la carrera")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Guardado: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Sensibilidad a temperatura
# ---------------------------------------------------------------------------

def plot_sensitivity(
    sensitivity_results: dict,
    tyre_life_range: np.ndarray,
    pit_loss: float,
    compound: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Curvas de tiempo proyectado para cada escenario de temperatura.
    Muestra como se desplaza (o no) la ventana optima.
    """
    _apply_style()

    n = len(sensitivity_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    cmap = plt.get_cmap("coolwarm")
    dt_vals = sorted(sensitivity_results.keys())
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for ax, (dt, color) in zip(axes, zip(dt_vals, colors)):
        res = sensitivity_results[dt]
        proj = res["projected_times"]
        opt  = res["optimal_lap"]

        ax.plot(tyre_life_range, proj, color=color, lw=2.2)
        ax.fill_between(tyre_life_range, proj, proj.min() - 0.05, alpha=0.08, color=color)

        if res.get("triggered", False):
            ax.axvline(opt, color="#00cc55", lw=1.5, linestyle="--")
            ax.set_title(f"dT = {dt:+.0f} °C\nOptimo: TL={opt}", fontsize=11)
        else:
            ax.set_title(f"dT = {dt:+.0f} °C\nSin ventana", fontsize=11)

        ax.set_xlabel("TyreLife")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    axes[0].set_ylabel("Tiempo de vuelta (s)")
    fig.suptitle(
        f"Sensibilidad a temperatura — {compound}  (pit_loss={pit_loss:.1f} s)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Guardado: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Importancia de features
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: RidgeCV,
    scaler: StandardScaler,
    feature_names: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Barras horizontales con importancia normalizada de cada feature:
    importancia[i] = |coef[i] * std[i]| / sum(...)
    """
    _apply_style()

    names = feature_names if feature_names is not None else FEATURE_NAMES
    importance = np.abs(model.coef_ * scaler.scale_)
    importance /= importance.sum()

    # Etiquetas legibles
    labels_map = {
        "TyreLife": "TyreLife",
        "TyreLife_sq": "TyreLife²",
        "TrackTemp": "TrackTemp",
        "TyreLife_x_TrackTemp": "TyreLife × TrackTemp",
        "AirTemp": "AirTemp",
        "Humidity": "Humidity",
        "WindSpeed": "WindSpeed",
    }
    labels = [labels_map.get(n, n) for n in names]

    pairs = sorted(zip(labels, importance), key=lambda x: x[1])
    labels_sorted, imp_sorted = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.25, 0.85, len(imp_sorted)))
    bars = ax.barh(labels_sorted, imp_sorted, color=colors, edgecolor="#cccccc", height=0.6)

    for bar, val in zip(bars, imp_sorted):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Importancia normalizada  (|coef × std|)")
    ax.set_title("Importancia de features del modelo de degradacion")
    ax.set_xlim(0, max(imp_sorted) * 1.18)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Guardado: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Backtesting vs realidad (consola)
# ---------------------------------------------------------------------------

def backtest_vs_reality(
    predicted_window: dict,
    real_pit_tyre_lives: list[int],
    compound: str,
    driver: str = "VER",
    year: int = 2021,
    gp: str = "Abu Dhabi",
) -> None:
    """
    Imprime en consola la comparacion entre la ventana predicha y el pit stop real.

    real_pit_tyre_lives: lista de TyreLife cuando el piloto entro realmente a boxes
                         (uno por pit stop en ese compuesto).
    """
    sep = "=" * 52
    print(sep)
    print(f"  BACKTESTING — {driver} {year} {gp} | {compound}")
    print(sep)
    print(f"  Ventana predicha : TL {predicted_window['window_start']} – "
          f"{predicted_window['window_end']}  "
          f"(optimo TL={predicted_window['optimal_lap']})")
    print(f"  Confianza        : +/-{predicted_window['confidence_laps']} vueltas")
    print(f"  Triggered        : {predicted_window['triggered']}")

    if real_pit_tyre_lives:
        print()
        for i, real_tl in enumerate(real_pit_tyre_lives, 1):
            diff = real_tl - predicted_window["optimal_lap"]
            inside = (predicted_window["window_start"] <= real_tl
                      <= predicted_window["window_end"])
            status = "DENTRO de la ventana" if inside else f"FUERA ({diff:+d} vueltas)"
            print(f"  Pit stop {i} real  : TL={real_tl}  -> {status}")
    else:
        print("  (sin pit stops reales proporcionados)")

    print(sep)
    logger.info(
        "Backtesting %s %d %s: predicho TL=%d, real %s",
        driver, year, compound,
        predicted_window["optimal_lap"],
        real_pit_tyre_lives,
    )
