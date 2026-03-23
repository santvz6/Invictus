# 🏛️ Arquitectura y Lógica del Dashboard INVICTUS

Este documento detalla el funcionamiento interno de la interfaz de usuario (`app.py`), los componentes visuales que la acompañan, cómo se estructuran los datos del GAP turístico, y cuál es la lógica matemática detrás de la detección de anomalías.

---

## 1. Estructura de Archivos del Dashboard

El panel de control está diseñado de forma modular utilizando **Streamlit**, separando la lógica central, la carga de datos y los elementos visuales.

### 📄 `dashboard/app.py` (Archivo Principal)
Es el núcleo de la aplicación. Aquí se encuentra la configuración inicial de la página (`st.set_page_config`) y el flujo visual general.
**Responsabilidades clave:**
- **Estado de Sesión (`st.session_state`)**: Mantiene en memoria variables importantes que no deben resetearse si el usuario hace clic o interactúa, como el `barrio_seleccionado`.
- **Carga Global de Datos**: Llama a `data_loader.py` para cargar los datasets procesados y cacheados.
- **Barra Lateral (Sidebar)**: Muestra los controles globales y filtros dinámicos (Mes de inicio/fin, Variable a mapear, Filtro por tipo de uso "Doméstico").
- **Agregación Dinámica**: Toma el dataset a nivel mensual y, dependiendo del rango de meses que seleccione el usuario, calcula la **media o el máximo** de cada variable agrupádolo por Barrio. Así, el barrio obtiene un color acorde a lo filtrado en ese rango temporal.
- **Lanzamiento de Componentes**: Renderiza los módulos separados (Mapa, Panel de anomalías, Informe inspector LLM) pasándoles los datos ya filtrados.

### 📚 `dashboard/data_loader.py`
Se encarga de la extracción de datos desde los CSV generados por el pipeline principal (`main.py`).
- Usa el decorador `@st.cache_data` para cargar los datos pesados en RAM **una sola vez** al arrancar.
- Realiza comprobaciones preventivas: Si no encuentra el CSV real procesado por el modelo de IA (`AMAEM-2022-2024_scaled.csv`), inyecta un DataFrame sintético (*mock*) para que la aplicación no falle abruptamente y pueda mostrarse algo (hasta que ejecutes el motor).

### 🧩 `dashboard/components/` (Directorio de Lógica Visual)
- **`map_view.py`**: Utiliza **Folium** o **Pydeck** para dibujar el mapa interactivo sobre las coordenadas de Alicante. Transforma el GeoJSON de los distritos en polígonos interactivos. Cruza los nombres de los barrios de `.geojson` con los nombres del `.csv` y dibuja el mapa de calor (Choropleth).
- **`anomaly_panel.py`**: Se renderiza *solo si* el usuario hace clic en un barrio. Muestra una vista en profundidad de sus métricas, incluyendo gráficos Plotly sobre el histórico de consumo.
- **`whatif_simulator.py`**: Interfaz técnica para cargar el modelo de `RandomForest` y modificar variables externamente mediante *sliders* para ver el impacto predictivo de "qué pasaría si aumenta el turismo".
- **`llm_report.py`**: Construye el contexto real del barrio seleccionado y se conecta a una IA Local (Ollama) para inyectarle un *prompt* que asume un rol directivo/inspector. Da como resultado el informe causal del Fraude.

---

## 2. 🔍 ¿Cómo y Dónde se detecta el GAP Turístico?

El concepto central del proyecto turístico radica en entender la gran bolsa de "viviendas fantasma".

### El Origen de la Discrepancia (El GAP)
1.  **Registro Abierto (GVA)**: Tienes el Registro Autonómico Oficial de Viviendas Turísticas operativas de la Generalitat Valenciana ("Lo que es legal y conocido").
2.  **Estimación Estadística Privada (INE - Exp. de Turismo)**: Tienes las plataformas de P2P scraping, censos telefónicos, e informes que realiza el Instituto Nacional de Estadística ("Lo que ocurre realmente, sumando legal e ilegal").

### Cálculo Algorítmico y Lógico
En el pipeline de datos previos (no en el dashboard web, sino en los `.ipynb` / `features_pipeline`), se cruzan temporalmente los registros de altas activas de GVA con los de INE.
- **GAP Absoluto** = `Num Viviendas Turísticas (Estimadas INE)` - `Num Alta Oficial (Comunidad GVA)`
- El Pipeline traslada y promedia este número a nivel "Barrio" usando el diccionario estadístico de pesos `mapping_barrios`.
- **En el Dashboard (`app.py`)**: Esa variable final llega calculada como `DatasetKeys.PCT_VT_SIN_REGISTRAR`. 

### Interpretación Visual
El mapa permite pintar intensidades según ese % GAP turístico, lo que sirve al ayuntamiento para enviar inspectores físicos **primero** a las zonas rojas del mapa antes de analizar si roban agua o no.

---

## 3. 🚨 La Detección Dual de Anomalías Acopladas (El Motor)

El fraude no puede deducirse por "quien consume más, defrauda", ya que si el barrio es un barrio rico de chalets en verano (Playa San Juan), es normal un altísimo consumo. Para ser una *Anomalía de Riesgo*:
El consumo **es atípicamente diferente del patrón esperado en ese bloque**. Y no solo eso, ocurre en una zona donde es propenso el blanqueo del P2P.

Para ello cruzamos **DOS redes neuronales**: Un modelo Físico (Fourier / Predictor) y un modelo Profundo (LSTM-Autoencoder) que evalúa toda la historia de telelectura del suministro.

### Paso 1: Predicción Base de lo "Normal"
- **Modelo Predictivo de Series Temporales (Fourier / Random Forest):** 
  Estudia cómo afecta la Estacionalidad (pico de verano, Fourier), el número de altas activas y el clima actual (Llueve / Hace calor). Y estima una *Predicción Matemática Física Segura* (El consumo debería de ser de *X* litros de media por hogar este mes).
  - En la gráfica de detalle de `anomaly_panel.py`, esto se marca como la **"Línea de Consumo Esperado (Físico)"**.

### Paso 2: Análisis Profundo del Patrón Cíclico (LSTM-Autoencoder)
- Una persona que destina su hogar a alquilar en Airbnb ilegal (un subarriendo), rompe la morfología de consumo para la que está clasificada el contrato domiciliario: en vez de consumos diarios en picos de mañana/tarde, existen vacíos temporales o crecimientos drásticos.
- **Autoencoder LSTM**: Esta es una red neuronal capaz de recordar la historia. Le inyectamos la secuencia cruda (Temp. del último año) y le pedimos a la propia red que *vuelva a escupir la misma información* por la salida.
- **El Error de Reconstrucción**: Si una vivienda se comportaba como una casa, la red memoriza su compresión como "casa", si de repente en julio se vuelve una piscina P2P con alta afluencia, los pesos de la red son incapaces de descifrarla correctamente en la capa latente y estallará produciendo un gran margen de error estadístico. *Esto es el `Reconstruction Error`*.
- En un barrio residencial puro de interior que siempre es estable, la red adivina la serie de forma exacta (Error = 0.01).
- En el "Casco Antiguo" en Julio, al romperse la topología habitual residencial, la reconstrucción fracasa y el *Error de Reconstrucción* se dispara. Esa es tu detección principal.

### Paso 3: Intersección con el Negocio (¿Es Fraude Turístico?)
Si el `Error de Reconstrucción` es enorme, hay una anomalía indudable técnica. Pero **no todas las anomalías son blanqueo/fraude de viviendas sin licencia**. Podría ser que hay una fuga gigantesca, o se abrió un gran parque húmedo.

*¿Cómo sabe INVICTUS clasificarlo a Turístico?*
Se formula un `FRAUD_RISK_SCORE` que solo brilla en `ROJO` cuando:
1.  (Matemático) El Ratio de Reconstrucción (Desajuste del patrón) ha superado el clúster.
2.  (Causal) El uso reportado es "Doméstico".
3.  (Fraude Negocio) Existe una presión (un GAP alto) de turismo P2P no declarado coincidente con un alto factor hotelero zonal en alta demanda climática.

Esto genera la etiqueta booleana: `DatasetKeys.ALERTA_TURISTICA_ILEGAL`

### En `dashboard/app.py`:
Toda la lógica matemática anterior es calculada asíncronamente en los clústers del pipeline. El Dashboard es solo un explorador estadístico de ese conocimiento procesado.
El mapa usa `app.py` para visualizar por coropletas cuáles son los barrios con puntuaciones altísimas en esos scores, permitiendo detectar fraude ilegal domiciliario mediante la huella hídrica y de un simple vistazo.
