"""
llm_report.py
-------------
Contenedor placeholder para el Informe de Hallazgos generado por LLM.
Listo para conectar a cualquier API (OpenAI, Gemini, etc.) en el futuro.
"""

import time
import streamlit as st


# Informe de ejemplo hardcodeado por barrio (mock)
_INFORMES_MOCK = {
    "PLAYA SAN JUAN": """
**Análisis del Barrio: PLAYA SAN JUAN**

📌 **Resumen Ejecutivo:**
El barrio de Playa San Juan presenta una concentración anómala de consumo hídrico durante los meses de verano (junio–agosto), con picos que superan en un **187%** el consumo esperado según el modelo físico.

🔴 **Anomalías Detectadas:**
- **3 contratos** registran consumos superiores a 1.5× el umbral físico de manera consistente durante 6+ meses.
- El error de reconstrucción medio del LSTM-AE es de **0.312**, significativamente superior al umbral del cluster (0.18).

🏖 **Indicadores Turísticos:**
- El porcentaje de Viviendas Turísticas registradas (**38.2%**) no justifica por sí solo el volumen detectado.
- Se estima un GAP de **127 unidades** entre el registro GVA y la demanda hídrica real, sugiriendo actividad no declarada.

💡 **Recomendación:**
Cruzar los contratos de agua de alto consumo con el padrón municipal y el registro catastral. Priorizar inspección en las calles de mayor densidad de plataformas de alquiler vacacional.
""",
    "DEFAULT": """
**Análisis de Barrio**

📌 **Resumen Ejecutivo:**
El modelo LSTM-Autoencoder ha procesado las secuencias temporales de consumo de agua de este barrio y ha identificado patrones de comportamiento relevantes.

🔴 **Anomalías Detectadas:**
El error de reconstrucción promedio se sitúa dentro de los rangos normales para este clúster de comportamiento. No se han identificado violaciones graves de las restricciones físicas.

🏖 **Indicadores Turísticos:**
Los datos de Viviendas Turísticas registradas (GVA) están en línea con el consumo hídrico observado. No se detecta un GAP significativo respecto a las estimaciones INE.

💡 **Recomendación:**
Continuar el seguimiento temporal. El modelo recomienda revisión periódica en los meses de mayor demanda estacional.
""",
}


def render_llm_report(barrio: str | None = None):
    """
    Renderiza el panel de informe LLM.

    Parameters
    ----------
    barrio : str | None — Barrio seleccionado en el mapa
    """
    st.markdown("### 🤖 Informe de Hallazgos IA")
    st.markdown(
        "<small style='color:#888;'>El sistema enviará el contexto del barrio a un LLM "
        "y mostrará un informe narrativo de las causas de anomalía.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not barrio:
        st.info("👆 Selecciona un barrio en el mapa para generar el informe.")
        return

    # Contexto que se enviaría al LLM (visible para el usuario)
    with st.expander("📋 Contexto enviado al LLM (debug)", expanded=False):
        st.code(f"""
BARRIO: {barrio}
MODELO: LSTM-Autoencoder (Water2Fraud v1.0)
VARIABLES: consumo, consumo_teorico_fisica, reconstruction_error,
           ALERTA_TURISTICA_ILEGAL, num_vt_barrio, pct_vt_barrio,
           temperatura_media, precipitacion
PROMPT: "Analiza las anomalías del barrio {barrio}. Explica las causas 
         más probables del desvío entre consumo real y esperado, 
         considerando el contexto turístico de Alicante."
        """, language="yaml")

    # ── Botón de generación ─────────────────────────────────────────────
    btn_key = f"llm_btn_{barrio}"
    if st.button(f"✨ Generar Informe para {barrio}", key=btn_key, type="primary"):
        st.session_state[f"llm_generated_{barrio}"] = False
        with st.spinner("Consultando modelo de lenguaje..."):
            time.sleep(1.8)  # Simula latencia de API
        st.session_state[f"llm_generated_{barrio}"] = True

    # ── Mostrar informe ─────────────────────────────────────────────────
    if st.session_state.get(f"llm_generated_{barrio}", False):
        informe = _INFORMES_MOCK.get(barrio, _INFORMES_MOCK["DEFAULT"])

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
            """<div style="background:rgba(82,183,136,0.1); border:1px dashed #52b788;
            border-radius:6px; padding:8px 12px; font-size:11px; color:#52b788; margin-top:8px;">
            🔧 <b>Para activar LLM real:</b> Reemplaza la función <code>_INFORMES_MOCK</code> por una llamada a
            <code>openai.ChatCompletion.create()</code> o <code>google.generativeai.GenerativeModel()</code>
            pasando el contexto del barrio seleccionado.
            </div>""",
            unsafe_allow_html=True,
        )
