# PROMPT PARA AGENTE ANTIGRAVITY: DESARROLLO DE DASHBOARD INTERACTIVO

**Objetivo:** Actúa como un experto en Ingeniería de Datos y Visualización Frontend (Streamlit/Dash/React). Tu tarea es construir un mapa interactivo avanzado utilizando la lógica de análisis, modelos de detección de anomalías y preprocesamiento de datos definidos en los Notebooks del proyecto.

---

### 1. Visualización de Mapa de Calor Dinámico
- **Lógica de Calor:** Implementa un mapa base que renderice un `Heatmap` basado en la densidad o valor de las variables del dataset.
- **Selector de Features:** Crea una botonera o menú de selección de características (features). Al cambiar la feature, el mapa debe actualizar la intensidad del calor en tiempo real (ej. Consumo, Temperatura, Humedad, etc.).
- **Escalado:** Asegúrate de normalizar los valores para que el gradiente de color sea representativo en todas las variables.

### 2. Controles de Filtrado y Tiempo
- **Slider Temporal:** Añade un control deslizante que permita al usuario navegar por meses/años. El mapa y las gráficas deben reaccionar al cambio temporal inmediatamente.
- **Filtro por Contrato:** Incluye un buscador o dropdown para filtrar la visualización por un ID de contrato específico o mostrar la vista agregada de todos los contratos.

### 3. Panel de Detalle de Anomalías (Slide Menu)
- **Interacción por Barrio:** Al hacer clic en un polígono (barrio), debe abrirse un menú lateral (sidebar) con:
    - Nombre del barrio y KPIs principales.
    - Listado de anomalías detectadas en ese sector.
    - **Gráfico Comparativo:** Utiliza el código de los archivos del proyecto para graficar el **Consumo Real vs. Consumo Esperado** (modelo predictivo), resaltando los puntos de anomalía.

### 4. Simulador de Features (What-if Analysis)
- **Entrada de Usuario:** Crea sliders para las variables principales basados en sus rangos $[min, max]$.
- **Gráfica en Tiempo Real:** Genera una gráfica de simulación que muestre cómo variaría el consumo o la predicción si el usuario modifica manualmente los valores de las features (ej. "Si subo la temperatura 2 grados, ¿cuánto sube el consumo esperado?").

### 5. Contenedor para Informe LLM
- Reserva un espacio en la interfaz (placeholder) donde se mostrará un "Informe de Hallazgos". 
- **Lógica:** El sistema debe estar preparado para enviar el contexto del barrio seleccionado a un LLM y mostrar el texto resultante (resumen de causas de anomalías).

---

**Instrucciones Técnicas Adicionales:**
- Utiliza las rutas de archivos de datos actuales para cargar los DataFrames.
- Asegúrate de que el estado de los filtros se mantenga al navegar entre barrios.
- Si es posible, utiliza librerías de alto rendimiento para el renderizado del mapa (como Pydeck o Folium con plugins de calor).