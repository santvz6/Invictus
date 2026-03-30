# 🛡️ INVICTUS: Pipeline Físico + Clasificación Estadística de Alertas

Este documento es el **estado real y consolidado** de la arquitectura del sistema INVICTUS. Sustituye y supera el plan original de `CLAUDE.md` (que proponía un Autoencoder LSTM + Motor de Reglas de Negocio complejo).

---

## 📌 Dimensiones de Datos (Fuentes Integradas)

El pipeline integra **5 dimensiones** en un único DataFrame:

| # | Fuente | Variables Clave |
|---|--------|----------------|
| 1 | 🚰 **AMAEM** (Telelectura) | `consumo_ratio`, `num_contratos`, `uso` |
| 2 | 🌤️ **AEMET** (Climatología) | `temperatura_media`, `precipitacion` |
| 3 | 🛰️ **Sentinel** (Teledetección) | `ndvi_satelite` |
| 4 | 🏢 **GVA** (Turismo Oficial) | `num_viviendas_barrio_gva`, `plazas_viviendas_barrio_gva` |
| 5 | 📊 **INE** (Turismo Oculto) | `porcentaje_vt_sin registrar %`, `pernoctaciones_vt_prov` |

---

## 🎯 Pipeline de 3 Pasos (Estado: ✅ Implementado en `src/model.py`)

### Paso 1: Predicción Física de Estacionalidad (Fourier)
Cada segmento `[barrio x uso]` tiene su propia "huella dactilar" hídrica, aprendida con datos de **2022-2023** (sin data leakage sobre 2024). El ajuste utiliza una onda de Fourier de 2º orden con tendencia lineal:

```
prediccion_fourier(t) = (m·t + c) + a1·cos(ω·t) + b1·sin(ω·t) + a2·cos(2ω·t) + b2·sin(2ω·t)
```

- `ω = 2π/12` → Captura el ciclo anual del consumo.
- **Fallback:** Si el ajuste falla, se usa la media histórica del segmento.

### Paso 2: Corrección Contextual (Random Forest Regressor)
El residuo de Fourier se modela contra las variables exógenas mediante un **Random Forest** (entrenado también solo con 2022-2023, prediciendo 2024 en modo real):

- **Features usadas:** temperatura, precipitación, NDVI, % viviendas sin registrar (GAP turístico), % festivos + `prediccion_fourier` + dummies de `uso`.
- Se genera: `consumo_teorico_fisica = prediccion_fourier + impacto_exogeno`

### Paso 3: Semáforo Z-Score de 6 Niveles (⚠️ El Corazón del Sistema)
El **residuo final** (`consumo_ratio - consumo_teorico_fisica`) se normaliza por segmento `[barrio x uso]` usando Z-Score robusto. Esto lo hace comparable entre barrios de distinto tamaño.

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

#### ⚖️ Imputación de Causas (Columnas de Salida)
Cada alerta incluye el `%` de "culpa" de cada variable externa, calculado como el peso relativo de su propio Z-Score:

| Columna en CSV | Descripción |
|-----|-----|
| `pct_calor_frio` | Peso de la temperatura anómala |
| `pct_lluvia_sequia` | Peso de la precipitación anómala |
| `pct_vegetacion` | Peso del NDVI anómalo (riegos/jardines) |
| `pct_turismo` | Peso del GAP turístico ilegal (`% sin registrar`) |
| `pct_fiesta` | Peso de los festivos del periodo |
| `pct_causa_desconocida` | **Residuo sin explicar** ← Principal diana del fraude |

---

## 📤 Salidas del Pipeline (`main.py --run`)

Genera dos archivos CSV en `internal/processed/`:

| Archivo | Contenido |
|---|---|
| `AMAEM-2022-2024_fisicos.csv` | Checkpoint detallado: predicciones + residuos + alertas |
| `AMAEM-2022-2024_not_scaled.csv` | Dataset completo no escalado para el Dashboard |

---

## 🔗 Estado de Integración y Tareas Pendientes

| Componente | Estado |
|---|---|
| `src/model.py` — Fourier + RF + Z-Score | ✅ Implementado |
| `src/config/string_keys.py` — `DatasetKeys` | ✅ Implementado |
| `src/config/features.py` — `FeatureConfig.CAUSAS_EXOGENAS` | ✅ Implementado |
| `main.py` — Orquestador del pipeline | ✅ Implementado |
| `dashboard/app.py` — Filtro por `alerta_nivel` | ⏳ Pendiente |
| `AMAEM_FISICOS6.ipynb` — Unificación con `main.py` | ⏳ Análisis exploratorio (no reemplazar el pipeline) |

> [!IMPORTANT]
> El notebook `AMAEM_FISICOS6.ipynb` es una **herramienta de exploración/validación independiente**, no la implementación del sistema. El pipeline real que genera los datos del Dashboard **es `main.py`**. El notebook debe usarse para validar que los outputs de `main.py` son correctos, no para sustituirlo.

> [!NOTE]
> Para generar los archivos procesados, ejecutar en una terminal con el entorno virtual activado:
> ```bash
> python main.py --run
> ```

---

## ❌ Cambios respecto al `CLAUDE.md` original (Plan Abandonado)

| Propuesto en CLAUDE.md | Por qué se descartó |
|---|---|
| Autoencoder LSTM | Complejo de mantener; overkill para datos de ~36 meses por barrio |
| Motor de Reglas de Negocio (business_rules.py) | Reemplazado por el Z-Score estadístico, más robusto y trazable |
| `FRAUD_RISK_SCORE` (0-100) | Reemplazado por `alerta_nivel` (categórico) + porcentajes de causa |
| Veto por USO (score → 0) | El modelo ya segmenta por `[barrio x uso]`, no puntúa residencial vs. comercial igual |
