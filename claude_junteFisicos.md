# 🛡️ INVICTUS: Pipeline Físico + Clasificación Estadística de Alertas (SHAP)

Este documento representa el **estado consolidado y final** de la arquitectura del sistema INVICTUS, descartando el plan inicial (`CLAUDE.md`) y sustituyendo la lógica de Z-scores estáticos por **Inteligencia Artificial Explicable (XAI) mediante Valores SHAP**, asegurándonos de coincidir con lo ya visualizable y expuesto en el Dashboard interactivo. 

---

## 📌 Dimensiones de Datos (Fuentes Integradas)

El pipeline integra **5 dimensiones** en un único DataFrame:

| # | Fuente | Variables Clave |
|---|--------|----------------|
| 1 | 🚰 **AMAEM** (Telelectura) | `consumo_ratio`, `num_contratos`, `uso` |
| 2 | 🌤️ **AEMET** (Climatología) | `temperatura_media`, `precipitacion` |
| 3 | 🛰️ **Sentinel** (Teledetección) | `ndvi_satelite` |
| 4 | 🏢 **GVA** (Turismo Oficial) | `num_viviendas_barrio_gva` |
| 5 | 📊 **INE** (Turismo Oculto) | `porcentaje_vt_sin registrar %` |

---

## 🎯 Pipeline de 3 Pasos (Implementado en `src/model.py`)

### Paso 1: Predicción Física de Estacionalidad (Fourier)
Cada segmento `[barrio x uso]` tiene su propia "huella dactilar" hídrica, aprendida con datos de **2022-2023** (sin data leakage sobre 2024). El ajuste utiliza una onda de Fourier de 2º orden con tendencia lineal:

```
prediccion_fourier(t) = (m·t + c) + a1·cos(ω·t) + b1·sin(ω·t) + a2·cos(2ω·t) + b2·sin(2ω·t)
```

### Paso 2: Corrección Contextual (Random Forest Regressor)
El residuo de Fourier se modela contra las variables exógenas mediante un **Random Forest** (entrenado también solo con 2022-2023, prediciendo todos los escenarios de fraude potencial de 2024):

- **Features usadas:** temperatura, precipitación, NDVI, % viviendas sin registrar, + variables dummys de contexto.
- Se genera `consumo_teorico_fisica = prediccion_fourier + impacto_exogeno`.

### Paso 3: Triaje Z-Score y Atribución Causal (XAI - SHAP) ⚠️
El **residuo final** (`consumo_ratio - consumo_teorico_fisica`) se normaliza por segmento usando un Z-Score robusto.  
**Para determinar y cuantificar matemáticamente la causa de la anomalía, se confiere a la librería `shap`**: usando un `TreeExplainer`, se extrae la contribución real (en cuotas de consumo) que tuvo cada feature exógena sobre el residuo de esa métrica.

#### 🚦 Clasificación de Alertas
| Nivel | Emoji | Condición `z_error_final` |
|-------|-------|--------------------------|
| Exceso GRAVE | 🔴 | `> +2.5` |
| Exceso MODERADO | 🟠 | `+2.0 < z ≤ +2.5` |
| Exceso LEVE | 🟡 | `+1.5 < z ≤ +2.0` |
| **Sin Alerta** | ✅ | `-1.5 ≤ z ≤ +1.5` |
| Defecto LEVE | 💧 | `-2.0 ≤ z < -1.5` |
| Defecto MODERADO | 💠 | `-2.5 ≤ z < -2.0` |
| Defecto GRAVE | 🔵 | `< -2.5` |

#### ⚖️ Imputación de Causas (Métricas SHAP en %)
Cada alerta incluye el porcentaje final de "culpabilidad" asignado al impacto de SHAP. Todo impacto residual no explicable linealmente ni por los árboles de decisión recae sobre la **Causa Desconocida** (la varianza pura descontrolada), principal objetivo estadístico de nuestra investigación por fraude.

| Columna de Salida | Imputación (Atribuida por SHAP Explainer) |
|-----|-----|
| `pct_calor_frio` | Responsabilidad adjudicada a Temperatura |
| `pct_lluvia_sequia` | Responsabilidad adjudicada a Precipitación |
| `pct_vegetacion` | Responsabilidad adjudicada a cambios en NDVI |
| `pct_turismo` | Responsabilidad directa del volumen irregular de viviendas (`% sin registrar`) |
| `pct_causa_desconocida` | Carga residual (Sospechoso Primario de Pérdidas de Agua) |

---

## 📤 Salidas del Pipeline y Gestión de Riesgos

La ejecución del núcleo (`main.py --run`) inyecta dos salidas en la troncal `internal/processed/`:
1. `AMAEM-2022-2024_fisicos.csv` (Pipeline Master con SHAP, Residuos y Alertas)
2. `AMAEM-2022-2024_not_scaled.csv` (Raw Dataset desescalinado para visualización web)

**Filtrado de Riesgos:** Adicionalmente a estas subidas maestras, el pipeline ahora aísla y escinde todas las Alertas por umbral, almacenando reportes rápidos preparados para los analistas dentro del directorio **`internal/processed/riesgos/`**:
* `1_EXCESO_Grave.csv` (Revisión Urbana Inmediata)
* `2_EXCESO_Moderado.csv` ... etc

---

## 🔗 Estado Actual y Roadmap 

| Componente Arquitectónico | Avance |
|---|---|
| Lógica Base Físico + RF + Z-Score (`src/model.py`) | ✅ Completado |
| Re-Estructuración XAI SHAP (`AMAEM_FISICOS7` -> `model.py`) | ✅ Consolidado |
| Rutas Estáticas de Riesgos (`src/config/paths.py`) | ✅ Consolidado |
| **Interactive Front-End (`streamlit run dashboard/app.py`)** | ✅ Totalmente Operativo |
| Motor *What-If* y Panel de Simulación | ✅ Integrado |
| Reportes LLM Contextualizados | ✅ Integrado |

> [!NOTE]
> Para lanzar todo el sistema en un solo pase de ejecución (extracciones, SHAP, generación de los 6 CSVs de riesgo), ejecutar en entorno virtual: `python main.py --run`. El Dashboard reaccionará de manera responsiva a los resultados actualizados.
