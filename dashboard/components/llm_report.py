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

from src.config import DatasetKeys

# Informe de ejemplo hardcodeado por barrio (mock)
_INFORMES_MOCK = {
    "DEFAULT": """
**Análisis de Barrio**

**Resumen Ejecutivo:**
El modelo LSTM-Autoencoder ha procesado las secuencias temporales de consumo de agua de este barrio y ha identificado patrones de comportamiento relevantes.

**Anomalías Detectadas:**
El error de reconstrucción promedio se sitúa dentro de los rangos normales para este clúster de comportamiento. No se han identificado violaciones graves de las restricciones físicas.

**Indicadores Turísticos:**
Los datos de Viviendas Turísticas registradas (GVA) están en línea con el consumo hídrico observado. No se detecta un GAP significativo respecto a las estimaciones INE.

**Recomendación:**
Continuar el seguimiento temporal. El modelo recomienda revisión periódica en los meses de mayor demanda estacional.
"""
}


def _build_dynamic_prompt(barrio: str, df: pd.DataFrame) -> str:
    """Extrae las métricas reales del DataFrame para construir el prompt."""
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
    if DatasetKeys.ALERTA_TURISTICA_ILEGAL in df_b.columns: agg_dict[DatasetKeys.ALERTA_TURISTICA_ILEGAL] = 'max'
        
    df_monthly = df_b.groupby(DatasetKeys.FECHA).agg(agg_dict).reset_index()

    consumo_real = df_monthly[DatasetKeys.CONSUMO].sum()
    if esperado_col in df_monthly.columns:
        # Consumo Total m3 = sumatoria de (Ratio_mes * Contratos_mes)
        consumo_esperado = (df_monthly[esperado_col] * df_monthly[DatasetKeys.NUM_CONTRATOS]).sum()
    else:
        consumo_esperado = consumo_real * 0.95

    alertas = df_monthly[DatasetKeys.ALERTA_TURISTICA_ILEGAL].sum() if DatasetKeys.ALERTA_TURISTICA_ILEGAL in df_monthly.columns else 0
    vt_pct = df_b[DatasetKeys.PCT_VT_BARRIO_INE].mean() if DatasetKeys.PCT_VT_BARRIO_INE in df_b.columns else 0.0
    
    # --- Variables de contexto extra ---
    contratos_promedio = df_monthly[DatasetKeys.NUM_CONTRATOS].mean() if not df_monthly.empty else 0
    temp_media = df_b[DatasetKeys.TEMP_MEDIA].mean() if DatasetKeys.TEMP_MEDIA in df_b.columns else 0.0
    ilegal_pct = df_b[DatasetKeys.PCT_VT_SIN_REGISTRAR].mean() if DatasetKeys.PCT_VT_SIN_REGISTRAR in df_b.columns else 0.0
    
    if esperado_col in df_monthly.columns:
        df_monthly["desvio_mensual"] = df_monthly[DatasetKeys.CONSUMO] - (df_monthly[esperado_col] * df_monthly[DatasetKeys.NUM_CONTRATOS])
    else:
        df_monthly["desvio_mensual"] = df_monthly[DatasetKeys.CONSUMO] * 0.05
        
    mes_pico = "N/A"
    if not df_monthly.empty and df_monthly["desvio_mensual"].max() > 0:
        idx_max = df_monthly["desvio_mensual"].idxmax()
        mes_pico = df_monthly.loc[idx_max, DatasetKeys.FECHA].strftime("%Y-%m")
    
    contexto = (
        f"**Rol**: Eres 'Invictus', un asistente experto de IA para el HACKATHON AMAEM DE DATOS.\n"
        f"**Misión**: Redactar un reporte analítico que demuestre innovación al cruzar datos de telelectura con datos abiertos externos.\n\n"
        f"**CONTEXTO TÉCNICO - BARRIO {barrio} ({fecha_inicio} a {fecha_fin})**:\n"
        f"  - Contratos activos: {contratos_promedio:,.0f} | Clima medio: {temp_media:.1f}ºC\n"
        f"  - Turismo Oficial: {vt_pct:.1f}% | Estimación Ilegales (Gap): {ilegal_pct:.1f}%\n"
        f"  - Consumo Real: {consumo_real:,.0f} m3 | Físico Esperado: {consumo_esperado:,.0f} m3\n"
        f"  - Desvío Total: {consumo_real - consumo_esperado:,.0f} m3 (Pico detectado en: {mes_pico})\n"
        f"  - Alertas de Fraude IA (Meses): {int(alertas)}\n\n"
        f"**Instrucciones**: NO inventes datos. Usa MÁXIMO 3 PÁRRAFOS CORTOS y directos en formato Markdown:\n"
    )
    return contexto

def render_llm_report(barrio: str | None = None, df: pd.DataFrame = None):
    """
    Renderiza el panel de informe LLM.

    Parameters
    ----------
    barrio : str | None — Barrio seleccionado en el mapa
    df : pd.DataFrame | None — Datos filtrados para generar contexto
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
        st.code(prompt_text, language="markdown")

    # ── Botón de generación ─────────────────────────────────────────────
    btn_key = f"llm_btn_{barrio}"
    
    if st.button(f"✨ Generar Informe para {barrio}", key=btn_key, type="primary"):
        st.session_state[f"llm_generated_{barrio}"] = False
        st.session_state[f"llm_response_{barrio}"] = None
        st.session_state[f"llm_is_mock_{barrio}"] = True
        
        with st.spinner("Consultando modelo de lenguaje..."):
            # Intentamos conectar a una instancia de Ollama (Local) para cumplir con las Normas del Hackathon
            # (Confidencialidad de los datos y herramientas Gratuitas / Open Source)
            try:
                response = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": "llama3.2", "prompt": prompt_text, "stream": False},
                    timeout=90
                )
                if response.status_code == 200:
                    st.session_state[f"llm_response_{barrio}"] = response.json().get("response", "")
                    st.session_state[f"llm_is_mock_{barrio}"] = False
            except Exception as e:
                # Fallback silencioso si no hay servidor local, evitamos romper el flujo
                print(f"[OLLAMA ERROR] No se pudo generar el reporte para {barrio}: {e}")
                time.sleep(1.5)
                pass
                
        st.session_state[f"llm_generated_{barrio}"] = True

    # ── Mostrar informe ─────────────────────────────────────────────────
    if st.session_state.get(f"llm_generated_{barrio}", False):
        informe = st.session_state.get(f"llm_response_{barrio}")
        if not informe:
            informe = _INFORMES_MOCK.get(barrio, _INFORMES_MOCK["DEFAULT"])
            
        if st.session_state.get(f"llm_is_mock_{barrio}"):
            st.warning("⚠️ Servidor IA local no detectado. Mostrando reporte de respaldo sintético.")

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
            ">{informe}</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption(
            "⚠️ Este informe es generado por un LLM y puede contener imprecisiones. "
            "Verificar siempre con los datos primarios antes de tomar decisiones."
        )

        # Placeholder para integración real de API
        st.markdown(
            """<div style="background:rgba(255,255,255,0.05); border:1px dashed rgba(255,255,255,0.2);
            border-radius:6px; padding:8px 12px; font-size:11px; color:#aaa; margin-top:8px;">
            🔧 <b>Nota de Cumplimiento (Bases Hackathon):</b> Para respetar la Cláusula 13.1 (Confidencialidad) y 
            la Cláusula 6.1 (Herramientas Gratuitas), este sistema está configurado para conectarse a una red 
            neuronal local (Ollama) en el puerto 11434, garantizando que <b>ningún dato de telelectura</b> 
            salga del entorno local.
            </div>""",
            unsafe_allow_html=True,
        )
