"""
llm_report.py
-------------
Contenedor placeholder para el Informe de Hallazgos generado por LLM.
Listo para conectar a cualquier API (OpenAI, Gemini, etc.) en el futuro.
"""

import time
import requests
import pandas as pd
import streamlit as st
import logging

from src.config import DatasetKeys, AIConstants, OllamaLLM
logger = logging.getLogger(__name__)

# Informe de ejemplo hardcodeado por barrio (mock)
_INFORMES_MOCK = {
    "DEFAULT": """
**Análisis de Barrio**

**Resumen Ejecutivo:**
El sistema Invictus ha procesado los patrones de consumo de agua de este barrio mediante el análisis de estacionalidad física (Fourier) e impacto de factores exógenos (Random Forest).

**Anomalías Detectadas:**
El desvío respecto al consumo físico esperado se sitúa dentro de los rangos normales para este perfil de barrio. No se han identificado violaciones graves de las restricciones físicas o climáticas.

**Indicadores Turísticos y Ambientales:**
La presión turística real y los factores climáticos están en línea con el consumo hídrico observado.

**Recomendación:**
Continuar el seguimiento temporal. El modelo recomienda revisión periódica en los meses de mayor demanda estacional.
"""
}


def _build_dynamic_prompt(barrio: str, df: pd.DataFrame) -> str:
    """
    Construye el prompt analítico enriquecido con datos reales para el LLM.

    Args:
        barrio (str): Nombre del barrio a analizar.
        df (pd.DataFrame): Dataset filtrado con el contexto histórico y actual.

    Returns:
        str: Prompt estructurado listo para ser enviado a la API de inferencia.
    """
    if df is None or df.empty:
        return f"Analiza las anomalías del barrio {barrio}. Sin embargo, no hay datos disponibles en este momento."

    barrios_limpios = df[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()
    df_b = df[barrios_limpios == barrio].copy()
    
    if df_b.empty:
        return f"Analiza el barrio {barrio}. No se encontraron registros en el periodo seleccionado."

    # 1. Filtrar por Uso Doméstico para coincidir con la lógica del panel de anomalías
    uso_col = DatasetKeys.USO
    if uso_col in df_b.columns and "DOMESTICO" in df_b[uso_col].values:
        df_b = df_b[df_b[uso_col] == "DOMESTICO"].copy()

    # 2. Contexto Temporal
    df_b[DatasetKeys.FECHA] = pd.to_datetime(df_b[DatasetKeys.FECHA])
    fecha_inicio = df_b[DatasetKeys.FECHA].min().strftime("%Y-%m")
    fecha_fin = df_b[DatasetKeys.FECHA].max().strftime("%Y-%m")

    # 3. Agrupación Mensual Matemática Correcta
    esperado_col = DatasetKeys.PREDICCION_FOURIER if DatasetKeys.PREDICCION_FOURIER in df_b.columns else DatasetKeys.CONSUMO_FISICO_ESPERADO
    
    agg_dict = { DatasetKeys.CONSUMO: 'sum', DatasetKeys.NUM_CONTRATOS: 'sum' }
    if esperado_col in df_b.columns: agg_dict[esperado_col] = 'mean'  # El físico devuelve Ratio (m3/contrato)
    if DatasetKeys.ALERTA_NIVEL in df_b.columns: agg_dict[DatasetKeys.ALERTA_NIVEL] = 'first'
        
    # Filtrar agg_dict por columnas que realmente existan en el DataFrame para evitar KeyError
    agg_dict = {k: v for k, v in agg_dict.items() if k in df_b.columns}
        
    df_monthly = df_b.groupby(DatasetKeys.FECHA).agg(agg_dict).reset_index()

    consumo_real = df_monthly[DatasetKeys.CONSUMO].sum()
    if esperado_col in df_monthly.columns:
        # Consumo Total m3 = sumatoria de (Ratio_mes * Contratos_mes)
        consumo_esperado = (df_monthly[esperado_col] * df_monthly[DatasetKeys.NUM_CONTRATOS]).sum()
    else:
        consumo_esperado = consumo_real * 0.95

    alertas = (df_monthly[DatasetKeys.ALERTA_NIVEL] != 'Normal').sum() if DatasetKeys.ALERTA_NIVEL in df_monthly.columns else 0
    niveles_detectados = df_monthly.loc[df_monthly[DatasetKeys.ALERTA_NIVEL] != "Normal", DatasetKeys.ALERTA_NIVEL].unique() if DatasetKeys.ALERTA_NIVEL in df_monthly.columns else []
    niveles_str = ", ".join(niveles_detectados) if len(niveles_detectados) > 0 else "Ninguno"
    
    # --- Variables de contexto extra ---
    temp_media = df_b[DatasetKeys.TEMP_MEDIA].mean() if DatasetKeys.TEMP_MEDIA in df_b.columns else 0.0
    precip_media = df_b[DatasetKeys.PRECIPITACION].mean() if DatasetKeys.PRECIPITACION in df_b.columns else 0.0
    ndvi_medio = df_b[DatasetKeys.NDVI_SATELITE].mean() if DatasetKeys.NDVI_SATELITE in df_b.columns else 0.0
    pernoct_media = df_b[DatasetKeys.PERNOCT_VT_PROV_INE].mean() if DatasetKeys.PERNOCT_VT_PROV_INE in df_b.columns else 0.0
    dias_festivos = df_b[DatasetKeys.DIAS_FESTIVOS].sum() if DatasetKeys.DIAS_FESTIVOS in df_b.columns else 0
    es_puente = df_b[DatasetKeys.ES_PUENTE].sum() if DatasetKeys.ES_PUENTE in df_b.columns else 0
    
    if esperado_col in df_monthly.columns:
        df_monthly["desvio_mensual"] = df_monthly[DatasetKeys.CONSUMO] - (df_monthly[esperado_col] * df_monthly[DatasetKeys.NUM_CONTRATOS])
    else:
        df_monthly["desvio_mensual"] = df_monthly[DatasetKeys.CONSUMO] * 0.05
        
    mes_pico = "N/A"
    if not df_monthly.empty and df_monthly["desvio_mensual"].max() > 0:
        idx_max = df_monthly["desvio_mensual"].idxmax()
        mes_pico = df_monthly.loc[idx_max, DatasetKeys.FECHA].strftime("%Y-%m")
    
    contexto = f"""Eres el sistema analítico de INVICTUS.
Analiza brevemente las anomalías hídricas del barrio {barrio} (periodo {fecha_inicio} a {fecha_fin}).
Eres un LLM que corre en local, así que tus respuestas deben ser muy concisas y estructuradas.

El modelo cruza:
1. Base Física (Fourier)
2. Impacto Exógeno (Random Forest): Clima, Nivel de Vegetación, Presión Turística (pernoctaciones reales) y Calendario (Festivos/Puentes).

DATOS DE ENTRADA:
- Consumo Real: {consumo_real:,.0f} m³
- Consumo Físico Esperado (Fourier): {consumo_esperado:,.0f} m³
- Mes con mayor desvío: {mes_pico}
- Alertas: {alertas} ({niveles_str})
- Temperatura Media: {temp_media:.1f} °C
- Precipitación Media: {precip_media:.1f} mm
- NDVI Satélite Medio: {ndvi_medio:.2f}
- Pernoctaciones Turísticas (Media): {pernoct_media:,.0f}
- Festivos en el periodo: {dias_festivos:.0f} (Días de puente: {es_puente:.0f})

INSTRUCCIONES DE FORMATO PARA TU RESPUESTA:
Actúa como un experto consultor en gestión del agua. Genera un reporte directo en formato Markdown usando esta estructura exacta (sin preámbulos):

### 🚨 Estado Cualitativo
(1 frase evaluando si la severidad de las alertas y el desvío mensual de consumo es preocupante o habitual).

### 🔍 Análisis Causal
(Un párrafo analizando qué variable generó la anomalía: ¿fue el clima, la presión turística real o los festivos? Menciona los datos de entrada provistos para justificarlo).

### 🛡️ Recomendación Operativa
(Una medida concreta y técnica de mitigación o monitoreo basada en la causa encontrada).

REGLA ESTRICTA: Sé directo, profesional e incisivo. No uses más de 150 palabras en toda tu respuesta."""
    return contexto

def render_llm_report(barrio: str | None = None, df: pd.DataFrame = None):
    """
    Renderiza la interfaz del informe de hallazgos IA en el dashboard.

    Args:
        barrio (str, optional): Nombre del barrio seleccionado.
        df (pd.DataFrame, optional): Datos filtrados para el contexto del informe.
    """
    st.markdown("### Informe de Hallazgos IA")
    st.markdown(
        "<small style='color:#888;'>El sistema enviará el contexto del barrio a un LLM "
        "y mostrará un informe narrativo de las causas de anomalía.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not barrio:
        st.info("Selecciona un barrio en el mapa para generar el informe.")
        return

    # Generamos el prompt de manera dinámica basado en los datos reales actuales del dashboard
    prompt_text = _build_dynamic_prompt(barrio, df)

    # Contexto que se enviaría al LLM (visible para el usuario)
    with st.expander("Contexto enviado al LLM (debug)", expanded=False):
        st.code("\n" + prompt_text, language="markdown")

    # ── Botón de generación ─────────────────────────────────────────────
    btn_key = f"llm_btn_{barrio}"
    
    if st.button(f"Generar Informe para {barrio}", key=btn_key, type="primary"):
        st.session_state[f"llm_generated_{barrio}"] = False
        st.session_state[f"llm_response_{barrio}"] = None
        st.session_state[f"llm_is_mock_{barrio}"] = True
        st.session_state[f"llm_error_{barrio}"] = None
        
        with st.spinner("Consultando modelo Qwen vía Ollama..."):
            # Inicializar cliente Ollama
            llm = OllamaLLM(model="qwen:7b", base_url="http://localhost:11434")
            
            # Verificar disponibilidad
            if not llm.health_check():
                error_msg = (
                    "**Ollama no está disponible**\n\n"
                    "Para usar esta función necesitas:\n"
                    "1. Descargar Ollama: https://ollama.ai\n"
                    "2. Ejecutar en terminal: `ollama pull qwen:7b`\n"
                    "3. Iniciar el servidor: `ollama serve`\n\n"
                    "Mientras tanto, mostrando reporte de respaldo..."
                )
                st.session_state[f"llm_error_{barrio}"] = error_msg
                st.session_state[f"llm_is_mock_{barrio}"] = True
            else:
                try:
                    # Generar usando Ollama
                    respuesta = llm.generate(prompt_text, stream=False, temperature=0.7)
                    st.session_state[f"llm_response_{barrio}"] = respuesta
                    st.session_state[f"llm_is_mock_{barrio}"] = False
                    logger.info(f"[OK] Informe generado exitosamente para {barrio}")
                except Exception as e:
                    error_msg = f"Error conectando con Ollama: {str(e)}"
                    logger.error(error_msg)
                    st.session_state[f"llm_error_{barrio}"] = error_msg
                    st.session_state[f"llm_is_mock_{barrio}"] = True
                
        st.session_state[f"llm_generated_{barrio}"] = True

    # ── Mostrar informe ─────────────────────────────────────────────────
    if st.session_state.get(f"llm_generated_{barrio}", False):
        # Verificar si hay error
        error_msg = st.session_state.get(f"llm_error_{barrio}")
        if error_msg:
            st.error(error_msg)
        
        informe = st.session_state.get(f"llm_response_{barrio}")
        if not informe:
            informe = _INFORMES_MOCK.get(barrio, _INFORMES_MOCK["DEFAULT"])
            
        if st.session_state.get(f"llm_is_mock_{barrio}"):
            st.info("Mostrando reporte de respaldo. Para usar Qwen, inicia Ollama.")

        st.markdown(
            f"""<div style="
                background: rgba(13,27,42,0.7);
                border: 1px solid rgba(76, 201, 240, 0.3);
                border-left: 4px solid #4cc9f0;
                border-radius: 8px;
                padding: 16px 20px;
                margin-top: 10px;
                font-family: 'Inter', sans-serif;
                line-height: 1.7;
                backdrop-filter: blur(6px);
            ">\n\n{informe}\n\n</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption(
            "Este informe es generado por un LLM y puede contener imprecisiones. "
            "Verificar siempre con los datos primarios antes de tomar decisiones."
        )

        # Placeholder para integración real de API
        st.markdown(
            """<div style="background:rgba(255,255,255,0.05); border:1px dashed rgba(255,255,255,0.2);
            border-radius:6px; padding:8px 12px; font-size:11px; color:#aaa; margin-top:8px;">
            <b>Nota de Cumplimiento (Bases Hackathon):</b> Para respetar la Cláusula 13.1 (Confidencialidad) y 
            la Cláusula 6.1 (Herramientas Gratuitas), este sistema está configurado para conectarse a una red 
            neuronal local (Ollama) en el puerto 11434, garantizando que <b>ningún dato de telelectura</b> 
            salga del entorno local.
            </div>""",
            unsafe_allow_html=True,
        )
