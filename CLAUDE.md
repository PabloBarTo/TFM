# F1 Strategy — Contexto completo para Claude Code

Este fichero contiene todo el diseño, decisiones y especificaciones del proyecto.
Léelo completo antes de escribir cualquier código.

---

## Objetivo del proyecto

Construir un modelo de degradación de neumáticos de F1 que calcule la **ventana óptima
de pit stop** para un piloto y carrera dados, teniendo en cuenta:

- La degradación mecánica del neumático (TyreLife)
- La temperatura del asfalto vuelta a vuelta (dinámica, no media)
- Variables climáticas completas (AirTemp, Humidity, WindSpeed)
- Datos históricos de varias ediciones del mismo GP para entrenar el modelo
- Validación cruzada por año (leave-one-year-out)
- Cálculo del pit stop loss real medido desde los datos, no hardcodeado

El caso de uso inicial es: **GP Abu Dabi 2021, Max Verstappen**.

---

## Entorno y estructura

```
f1-strategy/                  ← directorio raíz (ya creado por el usuario)
├── venv/                     ← entorno virtual Python (ya creado)
├── ff1_cache/                ← caché de FastF1 (no subir a git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← Capa 1
│   ├── weather.py            ← Capa 2
│   ├── model.py              ← Capa 3
│   ├── strategy.py           ← Capa 4
│   └── plots.py              ← Capa 5
├── main.py                   ← punto de entrada
├── requirements.txt
└── .gitignore
```

**Si el venv no está creado todavía**, créalo con:
```bash
cd "C:\Users\pablo\Desktop\Máster\TFM\f1-strategy"
python -m venv venv
venv\Scripts\activate
pip install fastf1 numpy pandas matplotlib scikit-learn scipy
pip freeze > requirements.txt
```

---

## Arquitectura: 5 capas

### Capa 1 — `src/data_loader.py`

**Responsabilidad:** cargar varias ediciones del mismo GP y unificarlas en un
DataFrame limpio listo para modelar.

**Variables FastF1 que usa:**

| Variable | Origen | Uso |
|---|---|---|
| `LapTime` | `session.laps` | Target Y — convertir a segundos |
| `TyreLife` | `session.laps` | Feature principal de degradación |
| `Compound` | `session.laps` | Segmentar modelo por compuesto |
| `IsAccurate` | `session.laps` | Filtro — solo True |
| `Deleted` | `session.laps` | Filtro — solo False |
| `TrackStatus` | `session.laps` | Filtro — solo "1" (pista verde) |
| `PitInTime` | `session.laps` | Filtro — NaT = vuelta limpia en pista |
| `PitOutTime` | `session.laps` | Filtro — NaT = vuelta limpia + medir pit loss |
| `LapStartTime` | `session.laps` | Join temporal con weather |
| `Time` (laps) | `session.laps` | Join temporal con weather (fin de vuelta) |
| `Stint` | `session.laps` | Agrupar vueltas por tanda |
| `FreshTyre` | `session.laps` | Feature — neumático nuevo al inicio del stint |
| `LapNumber` | `session.laps` | Ordenación y referencia |
| `Driver` | `session.laps` | Filtro por piloto |

**Lo que debe hacer `data_loader.py`:**

```python
def load_sessions(gp_name: str, years: list[int], session_type: str = "R") -> dict:
    """
    Carga varias ediciones del mismo GP.
    Devuelve dict {year: Session} con weather=True cargado.
    Usa ff1_cache/ para no re-descargar.
    """

def get_clean_laps(session, driver: str, compound: str = None) -> pd.DataFrame:
    """
    Filtra las vueltas válidas de un piloto:
    - IsAccurate == True
    - PitInTime es NaT  (no es vuelta de entrada a boxes)
    - PitOutTime es NaT (no es vuelta de salida de boxes)
    - TrackStatus == "1" (pista verde, sin SC ni bandera amarilla)
    - Deleted == False
    - Compound == compound (si se especifica)
    Añade columna LapTimeSec (LapTime en segundos).
    """

def measure_pit_loss(session, driver: str) -> float:
    """
    Calcula el pit stop loss neto real desde los datos:
    pit_loss = median(PitOutTime - PitInTime) - median_lap_time
    Mucho mejor que un valor hardcodeado.
    """

def build_training_dataframe(sessions: dict, driver: str, compound: str) -> pd.DataFrame:
    """
    Une las vueltas limpias de todas las ediciones en un solo DataFrame.
    Añade columna 'Year' para la validación cruzada.
    """
```

**Años a usar para Abu Dabi:** [2018, 2019, 2021, 2022, 2023]
(2020 fue en condiciones distintas por COVID, se puede incluir o excluir)

---

### Capa 2 — `src/weather.py`

**Responsabilidad:** enriquecer cada vuelta con la temperatura real que experimentó
el neumático, y construir una función de proyección de temperatura futura.

**Variables FastF1 que usa:**

| Variable | Origen | Uso |
|---|---|---|
| `TrackTemp` | `session.weather_data` | Feature más influyente. Asignar por join temporal. |
| `AirTemp` | `session.weather_data` | Feature secundaria |
| `Humidity` | `session.weather_data` | Feature — humedad relativa (%) |
| `WindSpeed` | `session.weather_data` | Feature — enfría el asfalto |
| `WindDirection` | `session.weather_data` | Opcional |
| `Rainfall` | `session.weather_data` | Filtro — excluir vueltas con lluvia |
| `Time` (weather) | `session.weather_data` | SessionTime de cada muestra (~cada 15 s) |

**Concepto crítico — join temporal:**

`weather_data` muestrea cada ~15 segundos. Cada vuelta dura 80-110 s, así que
hay 4-6 muestras por vuelta. El join correcto es:

```
Para cada vuelta:
    mask = (weather.Time >= lap.LapStartTime) AND (weather.Time <= lap.Time)
    TrackTemp_vuelta = mean(weather[mask]["TrackTemp"])
    # Fallback si no hay muestras: muestra más cercana a LapStartTime
```

**Lo que debe hacer `weather.py`:**

```python
def enrich_laps_with_weather(laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas TrackTemp, AirTemp, Humidity, WindSpeed a cada vuelta
    usando join temporal por rango [LapStartTime, Time].
    Elimina vueltas con Rainfall == True.
    """

def build_temp_forecast(weather_df: pd.DataFrame) -> callable:
    """
    Construye una función T_pred(session_time_seconds) -> float
    usando regresión lineal sobre los datos de weather_data observados
    hasta el momento actual de la carrera.

    En Abu Dabi (nocturna), la pista baja ~12°C de vuelta 1 a vuelta 58.
    Sin esto, el modelo se equivoca hasta 5 vueltas en la ventana óptima.

    Usar scipy.stats.linregress sobre weather_df["Time"] vs weather_df["TrackTemp"].
    """

def get_expected_temp_at_lap(lap_start_time, temp_forecast_fn) -> float:
    """
    Evalúa la función de proyección en un SessionTime futuro.
    Usado en la simulación de estrategia para predecir T en vueltas futuras.
    """
```

---

### Capa 3 — `src/model.py`

**Responsabilidad:** entrenar el modelo de degradación y exponerlo para predicción.

**Features de entrada al modelo:**

| Feature | Cómo se construye | Por qué |
|---|---|---|
| `TyreLife` | directa de laps | desgaste mecánico acumulado |
| `TyreLife²` | calculada: `TyreLife**2` | la degradación se acelera al final (cliff) |
| `TrackTemp` | del join temporal (capa 2) | asfalto caliente = más degradación |
| `TyreLife × TrackTemp` | calculada: producto | **el calor AMPLIFICA la degradación** — la feature más informativa |
| `AirTemp` | del join temporal | efecto sobre motor y aerodinámica |
| `Humidity` | del join temporal | humedad alta → ligeramente menos desgaste |
| `WindSpeed` | del join temporal | viento enfría asfalto → menos degradación |

**Target:** `LapTimeSec` (float, segundos)

**Algoritmo:** `RidgeCV` de scikit-learn con `alphas=[0.1, 1.0, 10.0, 100.0]`.
Por qué Ridge y no OLS: `AirTemp` y `TrackTemp` están correladas (multicolinealidad).
Ridge estabiliza los coeficientes con penalización L2.

**Validación:** leave-one-year-out. Entrenar con N-1 años, validar con el año
que se quiere predecir. Esto da un RMSE honesto porque el modelo nunca ha visto
ese año durante el entrenamiento.

```python
def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Construye la matriz X con las 7 features descritas arriba.
    Orden: [TyreLife, TyreLife², TrackTemp, TyreLife×TrackTemp, AirTemp, Humidity, WindSpeed]
    """

def train_model(df: pd.DataFrame) -> tuple[RidgeCV, StandardScaler]:
    """
    Entrena Ridge regression sobre el DataFrame completo (todas las ediciones).
    Devuelve (modelo_entrenado, scaler_ajustado).
    """

def cross_validate_by_year(df: pd.DataFrame) -> dict:
    """
    Leave-one-year-out cross validation.
    Para cada año en df["Year"].unique():
        - Entrenar con el resto de años
        - Predecir en el año excluido
        - Calcular RMSE y R²
    Devuelve dict {year: {"rmse": float, "r2": float}}
    """

def predict_lap_time(model, scaler, tyre_life: float, track_temp: float,
                     air_temp: float, humidity: float, wind_speed: float) -> float:
    """
    Predice el tiempo de vuelta para una combinación de condiciones.
    """

def predict_stint(model, scaler, tyre_life_range: np.ndarray,
                  temp_forecast_fn: callable, lap_start_times: np.ndarray,
                  avg_air: float, avg_humidity: float, avg_wind: float) -> np.ndarray:
    """
    Predice un array de tiempos de vuelta usando T(vuelta) dinámica,
    no una temperatura media fija.
    Esta es la diferencia clave respecto a modelos ingenuos.
    """
```

---

### Capa 4 — `src/strategy.py`

**Responsabilidad:** calcular cuándo parar usando los tiempos proyectados.

**Concepto central:**

```
Para cada vuelta candidata N de vida del neumático:
    coste_seguir = Σ [LapTime_pred(TL+i, T(vuelta+i)) - LapTime_pred(1, T(vuelta+i))]
                   (suma del tiempo extra pagado vs. neumático nuevo)

    Si coste_seguir >= pit_loss:
        → Es el momento de parar (ya hemos "pagado" el coste del pit stop)
```

La diferencia con el modelo anterior es que `T(vuelta+i)` es la temperatura
proyectada en esa vuelta futura concreta, no una media del stint.

```python
def compute_pit_window(projected_times: np.ndarray,
                       tyre_life_range: np.ndarray,
                       pit_loss: float) -> dict:
    """
    Calcula la ventana óptima de pit stop.
    Devuelve:
        {
            "optimal_lap": int,        # vida del neumático en el punto de cruce
            "window_start": int,       # optimal_lap - 2
            "window_end": int,         # optimal_lap + 2
            "delta_cumulative": np.ndarray,  # coste acumulado por vuelta
            "confidence_laps": int,    # ± basado en RMSE del modelo
        }
    """

def compute_undercut_window(my_projected_times: np.ndarray,
                             rival_stop_lap: int,
                             pit_loss: float,
                             tyre_life_range: np.ndarray) -> dict:
    """
    Si el rival para en vuelta R:
    ¿Cuántas vueltas más puedo aguantar yo antes de que pierda posición?

    Lógica:
    - El rival sale de boxes con neumático nuevo
    - Cada vuelta que yo sigo, el rival gana tiempo_nuevo vs mi tiempo_degradado
    - Cuando el rival me adelanta en tiempo acumulado → debo haber parado ya
    """

def sensitivity_analysis(model, scaler, tyre_life_range: np.ndarray,
                          base_conditions: dict,
                          temp_deltas: list[float],
                          pit_loss: float) -> dict:
    """
    ¿Cómo cambia la ventana óptima si la pista está X°C más caliente/fría?
    Devuelve dict {delta_t: {"optimal_lap": int, "projected_times": np.ndarray}}
    Para Abu Dabi: usar [-5, 0, +5, +10] °C respecto a la media observada.
    """
```

---

### Capa 5 — `src/plots.py`

**Responsabilidad:** visualización completa y backtesting.

```python
def plot_degradation(df_real, projected_times, tyre_life_range,
                     pit_window, compound, driver, session_info):
    """
    Gráfico principal:
    - Scatter de vueltas reales coloreadas por TrackTemp (cmap RdYlBu_r)
    - Línea del modelo proyectado
    - Banda verde sombreada = ventana óptima
    - Colorbar de temperatura
    """

def plot_delta_cumulative(delta_cum, tyre_life_range, pit_loss, optimal_lap):
    """
    Coste acumulado vs umbral del pit stop.
    Anotación en el punto de cruce.
    """

def plot_temperature_evolution(weather_df, pit_laps):
    """
    TrackTemp y AirTemp a lo largo de la carrera.
    Líneas verticales en los pit stops reales.
    """

def plot_sensitivity(sensitivity_results, pit_loss):
    """
    Un gráfico por escenario de temperatura.
    Muestra cómo se desplaza la ventana óptima.
    """

def plot_feature_importance(model, feature_names):
    """
    Barras horizontales con |coeficiente normalizado| de cada feature.
    """

def backtest_vs_reality(predicted_window, real_pit_laps, compound):
    """
    Imprime en consola la comparación:
    - Ventana predicha por el modelo
    - Pit stop real de Verstappen
    - Diferencia en vueltas
    """
```

---

### `main.py` — punto de entrada

```python
"""
Uso:
    python main.py --gp "Abu Dhabi" --year 2021 --driver VER --compound HARD
    python main.py --gp "Abu Dhabi" --year 2021 --driver VER --compound SOFT
"""

import argparse
from src.data_loader import load_sessions, get_clean_laps, measure_pit_loss, build_training_dataframe
from src.weather import enrich_laps_with_weather, build_temp_forecast
from src.model import train_model, cross_validate_by_year, predict_stint
from src.strategy import compute_pit_window, sensitivity_analysis
from src.plots import (plot_degradation, plot_delta_cumulative,
                       plot_temperature_evolution, plot_sensitivity,
                       plot_feature_importance, backtest_vs_reality)

TRAINING_YEARS = [2018, 2019, 2021, 2022, 2023]
```

El flujo en `main.py` debe ser:
1. Parsear argumentos (gp, year, driver, compound)
2. Cargar sesiones históricas (capa 1)
3. Para cada sesión: limpiar vueltas + enriquecer con weather (capa 2)
4. Construir DataFrame de entrenamiento unificado
5. Cross-validación por año → imprimir métricas (capa 3)
6. Entrenar modelo final con todos los años (capa 3)
7. Cargar la sesión objetivo (el año que queremos analizar)
8. Construir función de proyección de temperatura (capa 2)
9. Proyectar tiempos de vuelta con T(vuelta) dinámica (capa 3)
10. Calcular ventana óptima (capa 4)
11. Análisis de sensibilidad a temperatura (capa 4)
12. Generar todos los gráficos (capa 5)
13. Backtesting vs pit stop real (capa 5)

---

## Detalles importantes para Abu Dabi 2021

- **Carrera nocturna:** TrackTemp cae de ~38°C en la vuelta 1 a ~26°C en la vuelta 58.
  Sin temperatura dinámica, el modelo se equivoca hasta 5 vueltas.

- **Stints reales de Verstappen:**
  - Stint 1: MEDIUM (vueltas 1-13 aprox)
  - Stint 2: HARD (vueltas 14-35 aprox)
  - Stint 3: SOFT (vueltas 36-58 — la apuesta ganadora del campeonato)

- **Pit stop en vuelta 36:** fue posible gracias al safety car que entró en esa vuelta.
  El modelo no puede predecir el SC, pero sí que la ventana óptima de degradación
  ya se había alcanzado, lo que explica por qué Red Bull estaba listo para actuar.

- **Pit stop loss en Yas Marina:** históricamente ~19-21 segundos netos.
  Calcular desde los datos reales con `measure_pit_loss()`.

- **FastF1 nombre del GP:** usar `"Abu Dhabi"` (con h).

---

## Convenciones de código

- Docstrings en español
- Type hints en todas las funciones públicas
- Logging con `logging.getLogger(__name__)` en lugar de `print()`
- Constantes en MAYÚSCULAS al inicio de cada módulo
- Cada módulo es independiente y testeable por separado
- El caché de FastF1 siempre en `./ff1_cache/` relativo al directorio raíz

---

## Cómo empezar

```bash
# 1. Activar el entorno virtual
cd "C:\Users\pablo\Desktop\Máster\TFM\f1-strategy"
venv\Scripts\activate

# 2. Abrir Claude Code en este directorio
claude

# 3. Claude Code leerá este fichero automáticamente si está en la raíz
#    o puedes pasárselo explícitamente:
#    claude --context CLAUDE.md
```

---

## Orden de implementación recomendado

1. `src/data_loader.py` — sin esto no hay datos
2. `src/weather.py` — el join temporal es la pieza más crítica
3. `src/model.py` — con los dos anteriores ya se puede entrenar
4. `src/strategy.py` — usa las predicciones del modelo
5. `src/plots.py` — visualización de todo lo anterior
6. `main.py` — orquesta las 5 capas

Implementa y prueba cada módulo por separado antes de pasar al siguiente.
Para probar `data_loader.py` basta con cargar una sola sesión y verificar
que el DataFrame limpio tiene el número esperado de filas.
