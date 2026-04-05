# 🚀 Mejoras Propuestas — Proyecto INVICTUS

> Análisis técnico completo de áreas de mejora para el sistema Water2Fraud.
> Fecha de redacción: Abril 2026

---

## 1. 🧮 Simulador What-If (Alta prioridad)

### ❌ Problemas actuales
- **Beta simple (OLS lineal)**: El coeficiente `β = r × (σ_y / σ_x)` es una regresión de Pearson, incapaz de capturar las relaciones no-lineales entre, p.ej., temperatura extrema y consumo (el consumo no sube de forma lineal con el calor: hay umbrales físicos).
- **Cap fijo e independiente del barrio**: El tope de `±2.5σ / n_features` es el mismo para un barrio turístico costero que para un barrio residencial interior, aunque sus dinámicas sean radicalmente distintas.
- **Sin dimensión temporal**: Los sliders son agnósticos al mes. No se puede simular "¿qué pasa si en julio la temperatura sube 3°C?", solo "¿qué pasa si la temperatura media histórica sube 3°C?".
- **No usa el modelo RF existente**: El pipeline ya entrena un `RandomForestRegressor` con SHAP real. El What-If ignora ese modelo y recalcula sus propias betas desde cero.
- **Sin perfil anual**: Solo muestra un único valor de consumo simulado, sin mostrar cómo evolucionaría la curva de consumo durante los 12 meses.
- **Sin gestión de escenarios**: No se pueden guardar/comparar múltiples escenarios simultáneos.

### ✅ Mejoras implementadas (v2)
- **Betas por cuantil/percentil**: Se calcula la sensibilidad del consumo en tres zonas (frío/cálido, seco/húmedo, etc.) para detectar no-linealidad y usar el percentil de temperatura/precipitación seleccionado para elegir el beta correcto.
- **Modo Estacional**: Selector de mes de simulación. El delta se escala por la estacionalidad Fourier del barrio en ese mes específico.
- **Score de Plausibilidad del Escenario**: Calcula cuán probable es la combinación de features seleccionada usando distancia de Mahalanobis al centro histórico, mostrando un indicador "escenario plausible / atípico".
- **Radar chart de features**: Visualización polar que muestra los 5 features como % de desviación respecto a su media histórica, dando una lectura intuitiva del escenario completo.
- **Perfil mensual interpolado**: Si se selecciona un mes, el gráfico de comparativa muestra los 12 meses con el mes simulado "pinchado" en su posición real dentro de la curva anual.

---

## 2. 🗺️ Mapa de Calor Interactivo

### Mejoras a futuro
- **Animación temporal**: Añadir un slider de tiempo en el mapa para ver cómo evoluciona el estado de cada barrio mes a mes sin salir del mapa.
- **Capa de puntos de VT**: Superponer al mapa de calor los puntos GPS de las viviendas turísticas sin registrar.
- **Tooltips enriquecidos**: Añadir mini-sparkline SVG dentro del tooltip del mapa mostrando la evolución de Z-score.
- **Modo comparativo**: Split-view para comparar dos features simultáneamente (e.g., `consumo_ratio` vs `pct_vt_sin_registrar`).

---

## 3. 📊 Panel de Anomalías por Barrio

### Mejoras a futuro
- **Gráfico de Waterfall/Cascade**: En lugar del gráfico de factores conocidos en barras horizontales, un waterfall chart que muestre cómo cada factor *construye* el consumo simulado desde la base Fourier hasta el consumo real.
- **Comparativa entre barrios similares**: Añadir un botón "Ver barrios similares" que muestre la curva de consumo de los 3 barrios con perfil sociodemográfico más parecido.
- **Histograma de Z-scores**: Panel con distribución de Z-scores del barrio a lo largo del tiempo.

---

## 4. 🤖 Pipeline de Detección (Backend)

### Mejoras a futuro
- **Ventana deslizante para Fourier**: Re-entrenar Fourier con una ventana deslizante de N=18 meses para adaptarse automáticamente a cambios de tendencia.
- **Detección de changepoints**: Incorporar `ruptures` o `prophet` para detectar cambios estructurales.
- **Umbral adaptativo por barrio**: Los umbrales (1.5, 2.0, 2.5) podrían calcularse por barrio como cuantiles empíricos (e.g., `percentil_99` del barrio).
- **Ensemble de modelos**: Combinar el RF actual con ARIMA o Prophet.

---

## 5. 📋 Informe LLM

### Mejoras a futuro
- **Generación incremental (streaming)**: Usar `st.write_stream()` para mostrar el texto a medida que el LLM lo genera.
- **Descarga en PDF**: Botón para exportar el informe generado como PDF con logo INVICTUS y gráficos embebidos.
- **Plantillas configurables**: Permitir elegir entre diferentes plantillas de prompt (técnico, ejecutivo, jurídico).

---

## 6. 🔧 Calidad de Código y Arquitectura

### Mejoras a futuro
- **Tests unitarios**: Añadir tests con `pytest` y datos sintéticos conocidos.
- **Tipado estricto**: Completar las type hints y añadir `mypy` al CI.
- **Caché en session_state**: (Implementado en v2).

---

## Resumen de Prioridades

| Prioridad | Mejora | Impacto | Esfuerzo |
|-----------|--------|---------|----------|
| 🔴 Alta | What-If: betas no-lineales + modo estacional | Alto | Medio | (Hecho)
| 🔴 Alta | Cache de betas en `session_state` | Alto | Bajo | (Hecho)
| 🔴 Alta | Unificar umbrales Z en AIConstants | Medio | Bajo |
| 🟠 Media | Mapa: animación temporal | Alto | Alto |
| 🟠 Media | Panel: gráfico Waterfall | Medio | Medio |
| 🟠 Media | Informe LLM: streaming + PDF | Medio | Medio |
| 🟡 Baja | Tests unitarios | Medio | Medio |
| 🟡 Baja | Umbral adaptativo por barrio | Alto | Alto |
