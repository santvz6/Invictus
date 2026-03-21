"""
anomaly_panel.py
----------------
Panel lateral que se despliega al seleccionar un barrio en el mapa.
Muestra KPIs, gráfico comparativo Real vs Esperado y listado de anomalías.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch

from src.config import DatasetKeys, Paths

def _get_ae_reconstruction(df_full: pd.DataFrame, df_b: pd.DataFrame, barrio: str):
    """Intenta cargar el modelo Autoencoder y reconstruir la serie temporal real del barrio."""
    try:
        from src.water2fraud.models.autoencoder import LSTMAutoencoder
        
        cluster_id = df_b["cluster"].iloc[0] if "cluster" in df_b.columns else 0
        
        if not Paths.EXPERIMENTS_DIR.exists(): return None
        exp_dirs = sorted([d for d in Paths.EXPERIMENTS_DIR.iterdir() if d.is_dir()])
        if not exp_dirs: return None
        latest_exp = exp_dirs[-1]
        
        model_path = latest_exp / f"ae_cluster_{int(cluster_id)}.pth"
        if not model_path.exists(): return None
            
        scaled_csv = Paths.PROC_CSV_AMAEM_SCALED
        if not scaled_csv.exists(): return None
        df_scaled = pd.read_csv(scaled_csv)
        
        barrios_limpios_sc = df_scaled[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()
        df_b_sc = df_scaled[barrios_limpios_sc == barrio].copy()
        if df_b_sc.empty: return None
            
        df_b_sc[DatasetKeys.FECHA] = pd.to_datetime(df_b_sc[DatasetKeys.FECHA])
        df_b_sc = df_b_sc.sort_values([DatasetKeys.USO, DatasetKeys.FECHA])
        
        # Usamos el uso principal para no romper la secuencia LSTM
        uso_principal = "DOMESTICO" if "DOMESTICO" in df_b_sc[DatasetKeys.USO].values else df_b_sc[DatasetKeys.USO].iloc[0]
        df_b_sc = df_b_sc[df_b_sc[DatasetKeys.USO] == uso_principal]
        
        feature_cols = [
            DatasetKeys.CONSUMO_RATIO, DatasetKeys.MES_SIN, DatasetKeys.MES_COS,
            DatasetKeys.NUM_VT_BARRIO, DatasetKeys.PCT_VT_BARRIO, 
            DatasetKeys.OCUPACIONES_VT_PROV, DatasetKeys.PERNOCTACIONES_VT_PROV,
            DatasetKeys.CONSUMO_FISICO_ESPERADO, DatasetKeys.TEMP_MEDIA, 
            DatasetKeys.PRECIPITACION, DatasetKeys.NDVI_SATELITE
        ]
        feature_cols = [c for c in feature_cols if c in df_b_sc.columns]
        if not feature_cols: return None
        
        seq_len = 12
        group_data = df_b_sc[feature_cols].values
        fechas_data = df_b_sc[DatasetKeys.FECHA].values
        if len(group_data) < seq_len: return None
        
        model = LSTMAutoencoder(num_features=len(feature_cols), hidden_dim=128, latent_dim=16, seq_len=seq_len)
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        except TypeError: # Retrocompatibilidad con PyTorch antiguo
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        
        idx_ratio = feature_cols.index(DatasetKeys.CONSUMO_RATIO)
        min_val = df_full[DatasetKeys.CONSUMO_RATIO].min()
        max_val = df_full[DatasetKeys.CONSUMO_RATIO].max()
        if min_val == max_val: max_val = min_val + 1e-5
        
        fechas_reconst = []
        valores_reconst = []
        
        with torch.no_grad():
            for i in range(len(group_data) - seq_len + 1):
                seq = group_data[i : i + seq_len]
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                reconstruction = model(seq_tensor).squeeze(0).numpy()
                
                # Igual que el NoteBook: Reconstruimos la curva
                if i == 0:
                    for j in range(seq_len):
                        val_sc = reconstruction[j, idx_ratio]
                        valores_reconst.append(val_sc * (max_val - min_val) + min_val)
                        fechas_reconst.append(fechas_data[i + j])
                else:
                    val_sc = reconstruction[-1, idx_ratio]
                    valores_reconst.append(val_sc * (max_val - min_val) + min_val)
                    fechas_reconst.append(fechas_data[i + seq_len - 1])
                    
        return pd.DataFrame({DatasetKeys.FECHA: fechas_reconst, "ae_reconstruction": valores_reconst})
    except Exception as e:
        print(f"Error cargando Autoencoder en Dashboard: {e}")
        return None

def render_anomaly_panel(df: pd.DataFrame, barrio: str):
    # 1. Filtrar datos del barrio seleccionado (resolviendo prefijos como "01 - ")
    barrios_limpios = df[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()
    df_b = df[barrios_limpios == barrio].copy()
    
    st.markdown(f"### 🔎 Análisis: {barrio}")
    st.markdown("<hr style='margin: 0.5em 0; border-color: rgba(76,201,240,0.2);'/>", unsafe_allow_html=True)
    
    if df_b.empty:
        st.info("No hay datos disponibles para este barrio en el periodo seleccionado.")
        return

    # --- FIX: Agrupación temporal estricta ---
    # Agrupamos por mes para sumar todos los tipos de USO (Doméstico, Comercial...)
    # Esto elimina el efecto "zig-zag" (líneas salteadas) y arregla la escala del gráfico.
    esperado_col = DatasetKeys.PREDICCION_FOURIER if DatasetKeys.PREDICCION_FOURIER in df_b.columns else DatasetKeys.CONSUMO_FISICO_ESPERADO
    
    agg_dict = {
        DatasetKeys.CONSUMO: 'sum',
        DatasetKeys.NUM_CONTRATOS: 'sum',
    }
    if esperado_col in df_b.columns:
        agg_dict[esperado_col] = 'sum'
    if "ALERTA_TURISTICA_ILEGAL" in df_b.columns:
        agg_dict["ALERTA_TURISTICA_ILEGAL"] = 'max' # Si algún "uso" dio alerta, se marca el mes
    if "reconstruction_error" in df_b.columns:
        agg_dict["reconstruction_error"] = 'mean'
        
    df_monthly = df_b.groupby(DatasetKeys.FECHA).agg(agg_dict).reset_index()
    df_monthly = df_monthly.sort_values(DatasetKeys.FECHA)

    # 2. KPIs Principales
    consumo_total = df_monthly[DatasetKeys.CONSUMO].sum()
    esperado_total = df_monthly[esperado_col].sum() if esperado_col in df_monthly.columns else 0
    
    # Recalcular ratios limpios tras la agrupación
    contratos_safe = df_monthly[DatasetKeys.NUM_CONTRATOS].replace(0, 1)
    df_monthly["ratio_real"] = df_monthly[DatasetKeys.CONSUMO] / contratos_safe
    if esperado_col in df_monthly.columns:
        df_monthly["ratio_esperado"] = df_monthly[esperado_col] / contratos_safe
    else:
        df_monthly["ratio_esperado"] = 0.0

    # --- Integrar Reconstrucción del Autoencoder ---
    df_ae = _get_ae_reconstruction(df, df_b, barrio)
    usar_ae = False
    if df_ae is not None and not df_ae.empty:
        df_monthly = df_monthly.merge(df_ae, on=DatasetKeys.FECHA, how="left")
        df_monthly["ratio_esperado_final"] = df_monthly["ae_reconstruction"].fillna(df_monthly["ratio_esperado"])
        usar_ae = True
    else:
        df_monthly["ratio_esperado_final"] = df_monthly["ratio_esperado"]

    alertas = df_monthly["ALERTA_TURISTICA_ILEGAL"].sum() if "ALERTA_TURISTICA_ILEGAL" in df_monthly.columns else 0

    c1, c2 = st.columns(2)
    c1.metric("Consumo Total", f"{consumo_total:,.0f} m³", 
              delta=f"{(consumo_total - esperado_total):+,.0f} m³ vs Esperado", delta_color="inverse")
    c2.metric("Alertas Detectadas", int(alertas), 
              delta="CRÍTICO" if alertas > 0 else "Normal", delta_color="inverse" if alertas > 0 else "normal")

    # 3. Gráfico Comparativo: Ratio Real vs Esperado
    st.markdown("#### 📈 Ratio de Consumo Real vs. Esperado")
    
    fig = go.Figure()

    # Área base de Ratio Esperado (Añade una zona segura sombreada visualmente agradable)
    fig.add_trace(go.Scatter(
        x=df_monthly[DatasetKeys.FECHA], 
        y=df_monthly["ratio_esperado_final"],
        mode='lines', 
        name='Esperado (Autoencoder)' if usar_ae else 'Ratio Esperado',
        line=dict(color='#52b788', width=3, dash='dot'),
        fill='tozeroy',
        fillcolor='rgba(82, 183, 136, 0.1)',
        line_shape='spline' # Suavizado elegante
    ))

    # Línea de Ratio Real
    fig.add_trace(go.Scatter(
        x=df_monthly[DatasetKeys.FECHA], 
        y=df_monthly["ratio_real"],
        mode='lines+markers', 
        name='Ratio Real',
        line=dict(color='#4cc9f0', width=3),
        marker=dict(size=6, symbol='circle', color='#0d1b2a', line=dict(color='#4cc9f0', width=2)),
        line_shape='spline'
    ))

    # Resaltar puntos exactos de Anomalía
    if "ALERTA_TURISTICA_ILEGAL" in df_monthly.columns:
        df_anomalias = df_monthly[df_monthly["ALERTA_TURISTICA_ILEGAL"] > 0]
        if not df_anomalias.empty:
            fig.add_trace(go.Scatter(
                x=df_anomalias[DatasetKeys.FECHA], 
                y=df_anomalias["ratio_real"],
                mode='markers', 
                name='Anomalía Detectada',
                marker=dict(color='#ff4b4b', size=12, symbol='x', line=dict(width=2, color='#ff4b4b')),
                hovertext=df_anomalias["reconstruction_error"].apply(lambda x: f"Error AE: {x:.3f}"),
                hoverinfo="text+x+y"
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
        # FIX: Quitar rangemode="tozero" permite al gráfico hacer zoom a las fluctuaciones (no más líneas planas)
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="m³ / contrato", title_font=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        height=340,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Listado Tabular de Anomalías
    st.markdown("#### 🚨 Registro de Anomalías")
    if "ALERTA_TURISTICA_ILEGAL" in df_monthly.columns and alertas > 0:
        cols_to_show = [DatasetKeys.FECHA, "ratio_real", "ratio_esperado_final", "reconstruction_error"]
        cols_to_show = [c for c in cols_to_show if c in df_monthly.columns]
        
        df_table = df_anomalias[cols_to_show].copy()
        df_table[DatasetKeys.FECHA] = df_table[DatasetKeys.FECHA].dt.strftime("%Y-%m")
        
        # Formateo y renombrado visual
        df_table = df_table.rename(columns={
            DatasetKeys.FECHA: "Mes",
            "ratio_real": "Ratio Real",
            "ratio_esperado_final": "Esperado (AE)" if usar_ae else "Ratio Esperado",
            "reconstruction_error": "Error Autoencoder"
        })
        
        st.dataframe(df_table.style.format(precision=2), hide_index=True, use_container_width=True)
    else:
        st.success("✅ Sin comportamiento anómalo detectado en este periodo.")