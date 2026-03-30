import nbformat as nbf
import json

# Crear un nuevo notebook
nb = nbf.v4.new_notebook()

# Celda 1: Título (Markdown)
nb.cells.append(nbf.v4.new_markdown_cell(
    """# <center> **<span style="font-size:80px;">Físicos</span>** </center>
# 🌍 SISTEMA DE TRIAJE (6 NIVELES) CON DATOS REALES Y EXPORTACIÓN A CSV
---
Este notebook carga tus datos reales (CSV), calcula la predicción, asigna el porcentaje de causa (clima, turismo, etc.) a las anomalías y exporta automáticamente 6 archivos CSV clasificados por nivel de gravedad."""
))

# Celda 2: Importes y configuración
nb.cells.append(nbf.v4.new_code_cell(
    """import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import os

from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Cargar configuración del proyecto
sys.path.append(os.path.abspath(os.path.join("..")))
try:
    from src.config import DatasetKeys, Paths
    Paths.init_project()
    print("✅ Librerías y rutas del proyecto cargadas correctamente.")
except ImportError as e:
    print(f"❌ No se pudo cargar src.config: {e}")
    print("Asegúrate de estar en el directorio correcto.")"""
))

# Celda 3: Carga de datos
nb.cells.append(nbf.v4.new_code_cell(
    """# 1. CARGA DE DATOS REALES (Usando Paths)
print("1. Cargando tus archivos CSV reales desde Paths...")

try:
    # Cargar datos usando rutas centralizadas de Paths
    df_agua = pd.read_csv(Paths.PROC_CSV_AMAEM_NOT_SCALED)
    df_clima = pd.read_csv(Paths.AEMET_CLIMA_BARRIOS)
    df_ndvi = pd.read_csv(Paths.SENTINEL_NDVI)
    
    print(f"   ✅ Agua: {Paths.PROC_CSV_AMAEM_NOT_SCALED}")
    print(f"   ✅ Clima: {Paths.AEMET_CLIMA_BARRIOS}")
    print(f"   ✅ NDVI: {Paths.SENTINEL_NDVI}")
    
    # Normalizar nombres de columnas (minúsculas y sin espacios)
    df_agua.columns = df_agua.columns.str.lower().str.strip()
    df_clima.columns = df_clima.columns.str.lower().str.strip()
    df_ndvi.columns = df_ndvi.columns.str.lower().str.strip()
    
    # --- PREPARAR FECHA EN TODOS LOS DFs (meses cuadrados) ---
    # Agua: convertir 'fecha' a 'fecha_mes' si es necesario
    if 'fecha' in df_agua.columns and 'fecha_mes' not in df_agua.columns:
        df_agua['fecha_mes'] = pd.to_datetime(df_agua['fecha']).dt.to_period('M').astype(str)
    
    # --- AGRUPACIÓN CRÍTICA: Sumar todos los contratos por barrio (Evitar miles de falsas alertas e inflación gráfica) ---
    if 'consumo' in df_agua.columns:
        df_agua = df_agua.groupby(['fecha_mes', 'barrio'], as_index=False).sum(numeric_only=True)
    
    # Clima: convertir 'fecha' a 'fecha_mes' si es necesario
    if 'fecha' in df_clima.columns and 'fecha_mes' not in df_clima.columns:
        df_clima['fecha_mes'] = pd.to_datetime(df_clima['fecha']).dt.to_period('M').astype(str)
    
    # CLIMA: Renombrar 'zona' a 'barrio' si es necesario (AEMET usa 'zona' pero nosotros usamos 'barrio')
    if 'zona' in df_clima.columns and 'barrio' not in df_clima.columns:
        df_clima = df_clima.rename(columns={'zona': 'barrio'})
    
    # NDVI: debe tener fecha_mes ya, pero lo aseguramos
    if 'fecha' in df_ndvi.columns and 'fecha_mes' not in df_ndvi.columns:
        df_ndvi['fecha_mes'] = pd.to_datetime(df_ndvi['fecha']).dt.to_period('M').astype(str)
    
    # --- MERGE DE DATOS ---
    # Unimos los datos usando 'fecha_mes' y 'barrio' como conectores
    df_final = pd.merge(df_agua, df_clima, on=['fecha_mes', 'barrio'], how='left')
    # NDVI va a nivel global (sin barrio), así que lo mergeamos solo por fecha_mes
    df_final = pd.merge(df_final, df_ndvi, on=['fecha_mes'], how='left')
    
    # Rellenamos posibles huecos en blanco con 0 para que no falle la IA
    df_final = df_final.fillna(0)
    
    print(f"\\n✅ ¡Datos reales cargados y vinculados con éxito! Total de filas: {len(df_final)}")
    
except FileNotFoundError as e:
    print("\\n❌ ERROR: No encuentro alguno de los archivos.")
    print(f"Detalle: {e}")
    print("\\n⚠️ IMPORTANTE: El archivo procesado de AMAEM no existe. Debes ejecutar 'python main.py --run' en la terminal para que el backend local genere el dataset.")
except Exception as e:
    print(f"\\n❌ ERROR inesperado: {e}")
    import traceback
    traceback.print_exc()"""
))

# Celda 4: Fourier
nb.cells.append(nbf.v4.new_code_cell(
    """# 2. MOTOR MATEMÁTICO (FOURIER) 
print("\\n2. Calculando la estacionalidad física real...")
def modelo_fourier(t, m, c, a1, b1, a2, b2):
    w = 2 * np.pi / 12
    return (m * t + c) + (a1 * np.cos(w * t) + b1 * np.sin(w * t)) + (a2 * np.cos(2 * w * t) + b2 * np.sin(2 * w * t))

df_final['prediccion_fourier'] = 0.0
for barrio in df_final['barrio'].unique():
    mask = df_final['barrio'] == barrio
    y_barrio = df_final.loc[mask, 'consumo'].values
    t_barrio = np.arange(len(y_barrio))
    try:
        coef, _ = curve_fit(modelo_fourier, t_barrio, y_barrio, p0=[0, np.mean(y_barrio), 1000, 1000, 100, 100], maxfev=10000)
        df_final.loc[mask, 'prediccion_fourier'] = modelo_fourier(t_barrio, *coef)
    except:
        df_final.loc[mask, 'prediccion_fourier'] = np.mean(y_barrio)
print("✅ ¡Curvas calculadas!")"""
))

# Celda 5: Machine Learning
nb.cells.append(nbf.v4.new_code_cell(
    """# 3. MACHINE LEARNING 
print("\\n3. Entrenando IA global con tus datos...")
df_final['residuo'] = df_final['consumo'] - df_final['prediccion_fourier']
df_ml = pd.get_dummies(df_final, columns=['barrio'])

# Asegúrate de que estos nombres coinciden con las columnas de tus CSV
exogenas = ['tm_mes', 'p_mes', 'ndvi_satelite', 'num_vt_barrio', 'porcentaje_vt_barrio %', 'ocupaciones_vt_prov', 'pernoctaciones_vt_prov']

# Filtramos solo las variables exógenas que realmente existen en tu CSV por si falta alguna
exogenas_reales = [col for col in exogenas if col in df_final.columns]

columnas_barrios = [col for col in df_ml.columns if col.startswith('barrio_')]
X = df_ml[exogenas_reales + columnas_barrios]
y = df_ml['residuo']

ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
ml_model.fit(X, y)
df_final['impacto_exogeno'] = ml_model.predict(X)
df_final['prediccion_hibrida'] = df_final['prediccion_fourier'] + df_final['impacto_exogeno']
print("✅ ¡Entrenamiento completado!")"""
))

# Celda 6: Cálculo de causas y exportación
nb.cells.append(nbf.v4.new_code_cell(
    """# 4. CÁLCULO DE CAUSAS Y EXPORTACIÓN A 6 CSVs
print("\\n4. Calculando culpables y exportando archivos CSV...")

# --- CÁLCULO DE CULPAS (PORCENTAJES) ---
# Solo usamos las variables que existen en tus datos reales
sospechosos_posibles = {
    'tm_mes': 'Calor_Frio',
    'p_mes': 'Lluvia_Sequia',
    'ndvi_satelite': 'Estado_Jardines',
    'pernoctaciones_vt_prov': 'Turismo'
}
sospechosos = {k: v for k, v in sospechosos_posibles.items() if k in df_final.columns}

for var in sospechosos.keys():
    df_final[f'z_{var}'] = df_final.groupby('barrio')[var].transform(lambda x: stats.zscore(x, ddof=1)).fillna(0)
    df_final[f'peso_{var}'] = df_final[f'z_{var}'].abs()

df_final['error_final'] = df_final['consumo'] - df_final['prediccion_hibrida']
df_final['z_error_final'] = df_final.groupby('barrio')['error_final'].transform(lambda x: stats.zscore(x, ddof=1)).fillna(0)
df_final['peso_Desconocido'] = df_final['z_error_final'].abs()

columnas_pesos = [f'peso_{var}' for var in sospechosos.keys()] + ['peso_Desconocido']
df_final['suma_pesos'] = df_final[columnas_pesos].sum(axis=1)
df_final['suma_pesos'] = df_final['suma_pesos'].replace(0, 1) # Evitar divisiones por cero

columnas_porcentajes = []
for var, nombre_bonito in sospechosos.items():
    col_name = f'%_{nombre_bonito}'
    df_final[col_name] = (df_final[f'peso_{var}'] / df_final['suma_pesos']) * 100
    columnas_porcentajes.append(col_name)
    
df_final['%_Causa_Desconocida'] = (df_final['peso_Desconocido'] / df_final['suma_pesos']) * 100
columnas_porcentajes.append('%_Causa_Desconocida')

# --- FORMATEO VISUAL Y EXPORTACIÓN ---
df_final['consumo'] = df_final['consumo'].round(0)
df_final['prediccion_hibrida'] = df_final['prediccion_hibrida'].round(0)
df_final['z_error_final'] = df_final['z_error_final'].round(2)

columnas_vista = ['fecha_mes', 'barrio', 'consumo', 'prediccion_hibrida', 'z_error_final'] + columnas_porcentajes

# UMBRALES DEL SEMÁFORO
z_leve = 1.5
z_mod = 2.0
z_grave = 2.5

# --- FILTROS ---
exc_grave = df_final[df_final['z_error_final'] > z_grave][columnas_vista].copy()
exc_mod = df_final[(df_final['z_error_final'] > z_mod) & (df_final['z_error_final'] <= z_grave)][columnas_vista].copy()
exc_leve = df_final[(df_final['z_error_final'] > z_leve) & (df_final['z_error_final'] <= z_mod)][columnas_vista].copy()

def_grave = df_final[df_final['z_error_final'] < -z_grave][columnas_vista].copy()
def_mod = df_final[(df_final['z_error_final'] < -z_mod) & (df_final['z_error_final'] >= -z_grave)][columnas_vista].copy()
def_leve = df_final[(df_final['z_error_final'] < -z_leve) & (df_final['z_error_final'] >= -z_mod)][columnas_vista].copy()

# Formateamos los porcentajes con el símbolo '%'
tablas = [exc_grave, exc_mod, exc_leve, def_grave, def_mod, def_leve]
nombres_archivos = [
    '1_EXCESO_Grave.csv', '2_EXCESO_Moderado.csv', '3_EXCESO_Leve.csv', 
    '4_DEFECTO_Grave.csv', '5_DEFECTO_Moderado.csv', '6_DEFECTO_Leve.csv'
]

for i, tabla in enumerate(tablas):
    for col in columnas_porcentajes:
        tabla[col] = tabla[col].round(1).astype(str) + '%'
    # Exportar a CSV
    tabla.to_csv(nombres_archivos[i], index=False)

print("\\n📊 RESUMEN DE ALERTAS ENCONTRADAS (Exportadas a CSV):")
print(f"🔴 Exceso GRAVE: {len(exc_grave)} casos detectados.")
print(f"🟠 Exceso MODERADO: {len(exc_mod)} casos detectados.")
print(f"🟡 Exceso LEVE: {len(exc_leve)} casos detectados.")
print(f"🔵 Defecto GRAVE: {len(def_grave)} casos detectados.")
print(f"💠 Defecto MODERADO: {len(def_mod)} casos detectados.")
print(f"💧 Defecto LEVE: {len(def_leve)} casos detectados.")

print("\\n✅ ¡Listo! Busca los 6 archivos CSV en la barra lateral izquierda de tu Visual Studio Code.")"""
))

# Celda 7: Gráfica
nb.cells.append(nbf.v4.new_code_cell(
    """# 5. GRÁFICA MULTI-NIVEL (Opcional)
barrios_disponibles = df_final['barrio'].unique()
barrio_ejemplo = barrios_disponibles[0] # Coge el primer barrio de tu CSV automáticamente

df_plot = df_final[df_final['barrio'] == barrio_ejemplo].copy()
error_std = df_plot['error_final'].std()

plt.figure(figsize=(14, 6))
plt.fill_between(df_plot['fecha_mes'], 
                 df_plot['prediccion_hibrida'] - (error_std * 1.5), 
                 df_plot['prediccion_hibrida'] + (error_std * 1.5), 
                 color='green', alpha=0.1, label='Zona Segura (Z < 1.5)')

plt.plot(df_plot['fecha_mes'], df_plot['prediccion_hibrida'], 'k--', linewidth=2, alpha=0.7, label='Predicción IA')
plt.plot(df_plot['fecha_mes'], df_plot['consumo'], 'k-', label=f'Consumo Real ({barrio_ejemplo})', linewidth=2)

def pintar_alerta(df, mascara, color, marker, label, size=100):
    puntos = df[mascara]
    if not puntos.empty:
        plt.scatter(puntos['fecha_mes'], puntos['consumo'], c=color, marker=marker, s=size, label=label, zorder=5)

pintar_alerta(df_plot, df_plot['z_error_final'] > 2.5, 'red', 'o', '🔴 Exceso GRAVE', 150)
pintar_alerta(df_plot, (df_plot['z_error_final'] > 2.0) & (df_plot['z_error_final'] <= 2.5), 'orange', 'o', '🟠 Exceso MODERADO', 120)
pintar_alerta(df_plot, (df_plot['z_error_final'] > 1.5) & (df_plot['z_error_final'] <= 2.0), 'yellow', 'o', '🟡 Exceso LEVE', 90)

pintar_alerta(df_plot, df_plot['z_error_final'] < -2.5, 'blue', 'v', '🔵 Defecto GRAVE', 150)
pintar_alerta(df_plot, (df_plot['z_error_final'] < -2.0) & (df_plot['z_error_final'] >= -2.5), 'cyan', 'v', '💠 Defecto MODERADO', 120)
pintar_alerta(df_plot, (df_plot['z_error_final'] < -1.5) & (df_plot['z_error_final'] >= -2.0), 'lightblue', 'v', '💧 Defecto LEVE', 90)

plt.title(f'Auditoría {barrio_ejemplo}: Triaje de Anomalías por Niveles', fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Consumo de Agua')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""
))

# Guardar el notebook
with open('notebooks/Hugo.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✅ Notebook Hugo.ipynb creado correctamente!")
