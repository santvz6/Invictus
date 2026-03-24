# 🛡️ INVICTUS: Context-Aware Fraud Risk Scoring (Paso 3)

## 📌 Estado Actual del Proyecto
El pipeline (`main.py`) integra exhaustivamente datos de **5 dimensiones distintas**:
1. 🚰 **Telelectura (AMAEM):** Consumo, num_contratos, uso.
2. 🌤️ **Climatología (AEMET):** Temperatura media, precipitación.
3. 🛰️ **Teledetección (Sentinel):** NDVI (Vegetación).
4. 🏢 **Turismo Oficial (GVA):** Viviendas turísticas registradas.
5. 📊 **Turismo Oculto (INE):** Estimación real de viviendas turísticas (GAP o `PCT_VT_SIN_REGISTRAR`).

✅ **Modelado Físico (Fourier/Predictor):** Genera `CONSUMO_FISICO_ESPERADO` basado en clima.
✅ **Modelado Profundo (LSTM-Autoencoder):** Lee la historia y genera `RECONSTRUCTION_ERROR` cuando se altera el patrón cronológico.

❌ **Carencia Crítica (Falta del Paso 3):** 
Actualmente la IA detecta cualquier anomalía (Paso 1 y 2), pero el backend **no** cruza este error con AEMET, Sentinel ni el GAP (INE-GVA) para confirmar que se trata verdaderamente de **Fraude Turístico**.

---

## 🎯 Plan de Implementación: Motor de Scoring Contextual

El objetivo es crear un nuevo módulo (`src/water2fraud/models/business_rules.py`) que se ejecute al final del pipeline para convertir la matemática cruda en un `FRAUD_RISK_SCORE` accionable (0-100).

### 📐 Lógica del Scoring (Fórmula Híbrida de Atenuación y Amplificación)

```python
# Fórmula Conceptual
FRAUD_RISK_SCORE = Base_Risk * Business_Multiplier - Environmental_Discount
```

#### 1. Riesgo Base (IA Cruda)
El Autoencoder LSTM nos da el chivato principal:
* `Base_Risk` = Normalización de `RECONSTRUCTION_ERROR` (0 a 50 puntos). Si no hay ruptura del patrón de consumo, el score arranca en 0.

#### 2. Atenuación Física y Ambiental (AEMET + Sentinel)
Si el consumo sube anormalmente según el Autoencoder, pero el modelo Físico (instruido por AEMET) lo esperaba, *no es fraude turístico*.
* Si `CONSUMO_REAL <= CONSUMO_FISICO_ESPERADO`: El calor o la estacionalidad normal justifican el agua. **Discount masivo**.
* Si el índice vegetativo (`NDVI_SATELITE`) es muy alto en un barrio + mes sin lluvias (`PRECIPITACION` baja), el pico de agua inexplicada puede ser riego de jardines y piscinas privadas. **Discount parcial**.

#### 3. Amplificación por Negocio Turístico (INE vs GVA)
Si el desvío persiste tras la defensa ambiental (es un pico de agua que ni el calor ni la vegetación explican), buscamos el móvil del fraude.
* Aumentamos el riesgo residual multiplicándolo por el `PCT_VT_SIN_REGISTRAR` (GAP de mercado ilegal en ese barrio).
* Ampliamos también por `PCT_VT_BARRIO_INE` (si el barrio ya es un imán turístico, el contagio ilegal es alto).

#### 4. Veto por Uso
* Si `USO != DOMESTICO` $\rightarrow$ Score caee a **0**. (El fraude turístico persigue alquileres encubiertos domiciliarios, no podemos puntuar a un hotel comercial).

---

## 🛠 Tareas de Ejecución a programar

**Tarea 1: Crear `src/water2fraud/models/business_rules.py`**
- Implementar la función `compute_contextual_risk_score(df: pd.DataFrame) -> pd.DataFrame` encapsulando la lógica descrita arriba.
- Implementar `classify_risk_level(score: float)` para devolver `"🟢 BAJO", "🟡 MEDIO", "🟠 ALTO", "🔴 CRÍTICO"`.

**Tarea 2: Integrar en Pipeline (`main.py`)**
- Interceptar el DataFrame justo en el `Paso 5/6` de `main.py` antes de salvarlo como CSV sin escalar.
- Añadir las columnas definitivas al dataframe:
  - `DatasetKeys.FRAUD_RISK_SCORE`
  - `DatasetKeys.NIVEL_RIESGO`
- Eliminar los mocks y parcheos sintéticos generados en capas anteriores, el score debe provenir de cálculo verídico.

**Tarea 3: Actualizar `app.py` y `dashboard`**
- Asegurarse de quitar cualquier "truco visual" de `data_loader.py` para mapeos directos, y leer libremente las nuevas columnas reales que generará el Pipeline.

---
## 🏁 Hito
Una vez implementado, INVICTUS no solo "encontrará agua gastada", sino que presentará un listado justificado de barrios (basable en clima, vegetación y presión P2P) listos para que Urbanismo e Inspección emitan actas reales dirigidas.
