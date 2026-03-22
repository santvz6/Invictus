# Walkthrough: Dashboard Interactivo INVICTUS

## ¿Qué se construyó?

Dashboard **Streamlit** completo ([dashboard/app.py](file:///c:/Users/77422/Desktop/INVICTUS/dashboard/app.py)) que implementa los 5 módulos del [CLAUDE.md](file:///c:/Users/77422/Desktop/INVICTUS/CLAUDE.md), usando datos sintéticos como fallback inteligente cuando el pipeline aún no se ha ejecutado.

---

## Módulos implementados

| Archivo | Módulo CLAUDE.md | Estado |
|---|---|---|
| [dashboard/app.py](file:///c:/Users/77422/Desktop/INVICTUS/dashboard/app.py) | Orquestador principal | ✅ |
| [dashboard/data_loader.py](file:///c:/Users/77422/Desktop/INVICTUS/dashboard/data_loader.py) | Carga datos + datos mock | ✅ |
| [dashboard/components/map_view.py](file:///c:/Users/77422/Desktop/INVICTUS/dashboard/components/map_view.py) | §1 Mapa de Calor Dinámico | ✅ |
| [dashboard/components/anomaly_panel.py](file:///c:/Users/77422/Desktop/INVICTUS/dashboard/components/anomaly_panel.py) | §3 Panel de Anomalías (Slide Menu) | ✅ |
| [dashboard/components/whatif_simulator.py](file:///c:/Users/77422/Desktop/INVICTUS/dashboard/components/whatif_simulator.py) | §4 Simulador What-if | ✅ |
| [dashboard/components/llm_report.py](file:///c:/Users/77422/Desktop/INVICTUS/dashboard/components/llm_report.py) | §5 Contenedor Informe LLM | ✅ |

---

## Capturas de Pantalla (verificadas en navegador)

### 🗺 Tab 1: Mapa de Calor Interactivo

![Mapa de Calor con KPIs](C:\Users\77422\.gemini\antigravity\brain\aedc2373-32e2-4dc6-bf74-71be12c3d98e\main_page_metrics_map_1774104070162.png)

- **KPIs globales:** 30 barrios · 91,200 contratos · 9.4M m³ · 18 alertas activas
- **Mapa Folium** con heatmap choropleth y leyenda de gradiente de color
- **Sidebar** con filtro temporal (date picker), filtro por barrio y radio de features
- Panel derecho con prompt de clic en barrio para ver anomalías

### 🔬 Tab 2: Simulador What-if

![Simulador What-if](C:\Users\77422\.gemini\antigravity\brain\aedc2373-32e2-4dc6-bf74-71be12c3d98e\simulador_what_if_tab_1774104080323.png)

- Sliders para temperatura, precipitación, nº VTs y ratio consumo/contrato
- Estimación en tiempo real del consumo simulado + riesgo de anomalía
- Gráfico tornado de contribuciones y curva temperatura vs consumo

### 🤖 Tab 3: Informe LLM

![Informe LLM](C:\Users\77422\.gemini\antigravity\brain\aedc2373-32e2-4dc6-bf74-71be12c3d98e\informe_llm_tab_1774104089181.png)

- Prompt a usuario para seleccionar barrio en mapa
- Botón "Generar Informe" con spinner y mock de informe narrativo
- Expandible con el contexto YAML que se enviaría al LLM
- Guía inline de cómo conectar a OpenAI/Gemini real

---

## Grabación de verificación

![Dashboard Verification Recording](C:\Users\77422\.gemini\antigravity\brain\aedc2373-32e2-4dc6-bf74-71be12c3d98e\dashboard_verification_1774104041373.webp)

---

## Cómo lanzar

```powershell
# Desde la raíz del proyecto (con venv activado)
cd c:\Users\77422\Desktop\INVICTUS
.\venv\Scripts\streamlit.exe run dashboard/app.py
# → http://localhost:8501
```

> [!NOTE]
> Si el archivo `PROC_CSV_AMAEM_NOT_SCALED` no existe todavía, el dashboard genera **datos sintéticos realistas** automáticamente (30 barrios × 36 meses con estacionalidad y anomalías artificiales). Cuando ejecutes el pipeline real (`python main.py --run`), los datos reales se cargarán automáticamente al refrescar el dashboard.

---

## Próximos pasos sugeridos

1. **Conectar LLM real** en [llm_report.py](file:///c:/Users/77422/Desktop/INVICTUS/dashboard/components/llm_report.py) — reemplazar `_INFORMES_MOCK` por llamada a API
2. **Clic en polígono real** — mejorar la captura del barrio al clicar en el mapa GeoJSON cuando el pipeline genere el GeoJSON con los nombres exactos de barrio
3. **Ejecutar pipeline** — `python main.py --run` para poblar `PROC_CSV_AMAEM_NOT_SCALED` con datos reales
