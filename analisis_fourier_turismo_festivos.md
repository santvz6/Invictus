# 🔬 Análisis de Data Science: Turismo, Festivos y Fourier en INVICTUS

## Resumen ejecutivo

El pipeline actual tiene una **arquitectura híbrida sólida en teoría**, pero hay **problemas de diseño importantes** en cómo se tratan el turismo y los festivos respecto a Fourier. El clima (temperatura + precipitación) sí estás bien integrado. El turismo y los festivos tienen fallos estructurales que los degradan.

---

## 1. ¿Cómo funciona el sistema actual?

El modelo es un híbrido de dos etapas:

```
CONSUMO_REAL
    = PREDICCION_FOURIER       ← Estacionalidad pura (Barrio x Uso)
    + IMPACTO_EXOGENO          ← Lo que el RF aprende del residuo
    + RESIDUO_FINAL            ← Lo que ni Fourier ni el RF explican → Fraude
```

**El punto crítico:** Fourier se ajusta sobre `CONSUMO_RATIO` bruto, que **ya contiene dentro** el efecto del turismo y los festivos. Luego el RF intenta aprender esos efectos del residuo. Esto crea una **competencia interna** entre ambos componentes.

---

## 2. Variables Climáticas (AEMET) — ✅ BIEN TRATADAS

**Fichero:** `src/features/aemet_processor.py`

```python
# Cruce geo-temporal correcto: barrio + mes
df_final = pd.merge(df_final, df_aemet, on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'], how='left')
```

**¿Por qué funcionan bien?**
- `TEMP_MEDIA` y `PRECIPITACION` tienen correlación directa y lineal con el consumo
- Son variables **continuas y suavizadas mensualmente**
- Su variación es lenta → son quasi-ortogonales a la señal de Fourier
- El RF puede aprender su impacto en el residuo sin interferir con la onda anual

**Escala en features:** `MIN_MAX` → correcto para variables acotadas.

---

## 3. Turismo (INE + GVA) — ⚠️ PROBLEMAS SERIOS

### 3.1 ¿Qué variable usa el modelo?

**Fichero:** `src/config/features.py` (línea 46)

```python
PIPELINE_FEATURES = {
    DatasetKeys.PCT_VT_SIN_REGISTRAR: FeatureScaling.ROBUST  # ← Esta es la única que entra al RF
}
```

**Fichero:** `src/features/preprocessor.py` (líneas 115-119)

```python
# Estimación de ilegales = INE - GVA (clipeado a 0)
df[DatasetKeys.NUM_VT_SIN_REGISTRAR] = (
    df[DatasetKeys.NUM_VT_BARRIO_INE] - df[DatasetKeys.NUM_VT_BARRIO_GVA]
).clip(lower=0)
```

### 3.2 ❌ Problema 1: Se mide la "ilegalidad", no la presión turística

El modelo no usa `PCT_VT_BARRIO_INE` (total de VT reales) sino **el gap INE-GVA**, que mide pisos ilegales, NO el volumen de turistas en la ciudad. Esto está bien para detectar fraude fiscal, pero **NO para modelar el impacto en el consumo de agua**.

**La lógica correcta sería:**
- Más turistas → más duchas, más piscinas, más consumo en bares
- Eso depende de `OCUP_VT_PROV_INE` (alojamientos ocupados) o `PERNOCT_VT_PROV_INE` (noches)
- No de si los pisos están registrados o no

### 3.3 ❌ Problema 2: `OCUP_VT_PROV_INE` y `PERNOCT_VT_PROV_INE` se calculan pero NO entran al modelo

**Fichero:** `src/config/string_keys.py` (líneas 36-37)

```python
OCUP_VT_PROV_INE    = "ocupaciones_vt_prov"
PERNOCT_VT_PROV_INE = "pernoctaciones_vt_prov"
```

**Fichero:** `src/config/features.py` — estas claves **NO aparecen** en `PIPELINE_FEATURES` ni en `CAUSAS_EXOGENAS`.

**Conclusión:** Se cargan, se procesan, se guardan en el CSV intermedio... y luego se ignoran completamente.

### 3.4 ❌ Problema 3: El turismo tiene componente estacional propio que Fourier "roba"

El turismo en Alicante es fuertemente estacional (pico en julio-agosto). Como Fourier se ajusta a `CONSUMO_RATIO` bruto, aprende también esa curva. Cuando luego el RF intenta aprender el efecto del turismo del residuo, **ese efecto ya fue absorbido por la onda de Fourier**.

**Diagnóstico:** El `PCT_TURISMO` en las causas finales probablemente estará siempre bajo porque Fourier lo "robó".

---

## 4. Festivos (HolidayBarrioProcessor) — ⚠️ PROBLEMAS MODERADOS

### 4.1 ¿Qué variable usa el modelo?

**Fichero:** `src/config/features.py` (línea 43)

```python
DatasetKeys.PCT_FESTIVOS: FeatureScaling.MIN_MAX
```

**Fichero:** `src/features/holiday_barrio_processor.py` (líneas 78-83)

```python
df_festivos = df_festivos.rename(columns={
    'Dias_Festivos': DatasetKeys.DIAS_FESTIVOS,
    'Porcentaje_Anual': DatasetKeys.PCT_FESTIVOS  # Porcentaje anual del mes respecto al año
})
```

### 4.2 ❌ Problema 1: El porcentaje anual de festivos no captura el efecto real

`PCT_FESTIVOS` es el porcentaje de días festivos del mes sobre el año. Esto mide "concentración de festivos", no "impacto en el consumo". El problema:

- Semana Santa (marzo/abril) tiene festivos → consumo ↑ turismo interior + familias en casa
- Agosto tiene pocos festivos nacionales → pero el consumo sube por el calor y el turismo

La variable **no distingue entre tipos de festivo** (local, autonómico, nacional) y su efecto en el consumo puede ser opuesto según el barrio.

### 4.3 ❌ Problema 2: Granularidad mensual diluye el efecto real de los festivos

Los festivos son eventos de días concretos. Al agregar a mes, un mes con 2 días festivos seguidos (puente) y uno con 2 días festivos separados tienen el **mismo `PCT_FESTIVOS`**, pero el impacto en el consumo es muy diferente.

### 4.4 ⚠️ Problema 3: Misma trampa de Fourier

Al igual que el turismo, los festivos tienen un patrón anual (Semana Santa = siempre en marzo/abril, agosto siempre con feria local...). Fourier aprende ese patrón y el RF tiene poco residuo relacionado con festivos que aprender.

### 4.5 ✅ Lo que sí funciona

```python
# Cruce por barrio + mes: correcto, los festivos locales son distintos por barrio
df_final = pd.merge(df_final, df_festivos, on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'], how='left')
```

El hecho de que los festivos estén a nivel de barrio es correcto y valioso, porque la Feria de Agosto afecta más a ciertos barrios que a otros.

---

## 5. El Problema Central: Fourier absorbe la estacionalidad del turismo y los festivos

```
Fourier ajusta sobre: CONSUMO_RATIO (bruto)
                      ↑
                      Incluye: efecto del calor
                               efecto del turismo de verano
                               efecto de Semana Santa
                               efecto de feria de agosto
                               ...todo mezclado
```

Esto significa que **la curva de Fourier ya aprende indirectamente el turismo y los festivos**. El residuo que le queda al RF es casi ruido, y el RF apenas puede aprender nada útil sobre turismo/festivos.

**La prueba:** Si miramos `CAUSAS_EXOGENAS` en la salida, `PCT_TURISMO` y `PCT_FIESTA` tendrán valores SHAP muy bajos sistemáticamente.

---

## 6. Propuestas de Mejora

### Mejora 1 (Impacto Alto): Añadir métricas de presión turística real al modelo

**Fichero:** `src/config/features.py`

```python
# AÑADIR estas líneas al PIPELINE_FEATURES:
DatasetKeys.PCT_VT_BARRIO_INE:    FeatureScaling.MIN_MAX,   # Penetración turística total
DatasetKeys.OCUP_VT_PROV_INE:     FeatureScaling.MIN_MAX,   # Ocupación real provincial
DatasetKeys.PERNOCT_VT_PROV_INE:  FeatureScaling.ROBUST,    # Pernoctaciones reales

# Y en CAUSAS_EXOGENAS, cambiar PCT_VT_SIN_REGISTRAR por algo más representativo:
DatasetKeys.OCUP_VT_PROV_INE:  DatasetKeys.PCT_TURISMO,     # Más representativo que el GAP
```

### Mejora 2 (Impacto Alto): Usar `DIAS_FESTIVOS` en lugar de (o además de) `PCT_FESTIVOS`

`DIAS_FESTIVOS` es un conteo directo y tiene más poder predictivo. Además considerar añadir una variable binaria de "puente":

```python
# En holiday_barrio_processor.py, añadir Feature de Puente:
df_festivos['es_puente'] = (df_festivos['dias_festivos'] >= 2).astype(int)
```

### Mejora 3 (Impacto Crítico): Corregir el Input de Fourier para desacoplar la estacionalidad

En lugar de ajustar Fourier sobre el consumo bruto, **desestacionalizar primero** por las variables exógenas antes de ajustar Fourier. La arquitectura correcta sería:

```
Opción A (Fourier como residualizador):
  1. RF predice el efecto de turismo/festivos/clima sobre el consumo
  2. Fourier se ajusta al RESIDUO del RF (consumo - impacto_RF)

Opción B (Fourier sobre consumo "neutral"):
  1. Calcular un "consumo base" = consumo en meses de baja presión turística y sin festivos
  2. Fourier aprende SOLO la estacionalidad "natural" del agua
  3. RF aprende el delta turismo/festivos/clima sobre ese baseline
```

### Mejora 4 (Impacto Medio): Añadir features temporales que capturen los festivos mejor

```python
# En preprocessor.py o holiday_barrio_processor.py:
df['semana_santa'] = df['fecha'].apply(lambda d: 1 if d.month in [3, 4] else 0)
df['verano']       = df['fecha'].apply(lambda d: 1 if d.month in [6, 7, 8] else 0)
df['navidad']      = df['fecha'].apply(lambda d: 1 if d.month in [12, 1] else 0)
```

Estas features son **ortogonales a Fourier** porque son binarias (no sinusoidales) y el RF puede aprender sus efectos específicos.

---

## 7. Tabla de Diagnóstico

| Variable | ¿Entra al RF? | ¿Entra correcta? | ¿Fourier la "roba"? | Prioridad mejora |
|---|---|---|---|---|
| `TEMP_MEDIA` | ✅ Sí | ✅ Sí | ⚠️ Parcial | Baja |
| `PRECIPITACION` | ✅ Sí | ✅ Sí | ✅ No (no estacional) | Baja |
| `NDVI_SATELITE` | ✅ Sí | ✅ Sí | ⚠️ Parcial | Media |
| `PCT_VT_SIN_REGISTRAR` | ✅ Sí | ❌ Mide ilegalidad, no presión | ✅ No | Alta |
| `OCUP_VT_PROV_INE` | ❌ No | — | — | **Crítica** |
| `PERNOCT_VT_PROV_INE` | ❌ No | — | — | **Crítica** |
| `PCT_VT_BARRIO_INE` | ❌ No | — | — | Alta |
| `PCT_FESTIVOS` | ✅ Sí | ⚠️ Métrica débil | ❌ Sí | Alta |
| `DIAS_FESTIVOS` | ❌ No | — | — | Alta |

---

## 8. Conclusión

**Turismo:** La variable usada (`PCT_VT_SIN_REGISTRAR`) mide fraude fiscal, no presión sobre el consumo. Las variables útiles (`OCUP_VT_PROV_INE`, `PERNOCT_VT_PROV_INE`) se calculan pero se descartan. Esto es el error más importante del sistema.

**Festivos:** La métrica `PCT_FESTIVOS` (porcentaje anual) es débil. `DIAS_FESTIVOS` sería mejor. Ambas sufren la absorción por Fourier.

**Fourier:** Correcto para capturar el ciclo anual del agua, pero al ajustarse sobre el consumo bruto "contamina" su onda con efectos de turismo y festivos, empobreciendo lo que el RF puede aprender.
