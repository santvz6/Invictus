# Guía de Ejecución (INVICTUS)

Esta guía detalla los pasos multiplataforma (Windows, Ubuntu/Linux, macOS) para instalar, ejecutar y configurar el proyecto INVICTUS.

---

## 1. Instalación y Entorno Base

### Requisitos Previos:
- **Python 3.13+** instalado.
- **Git** instalado.

### Pasos de Instalación:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/santvz6/Invictus.git
   cd Invictus
   ```

2. **Crear y activar el entorno virtual:**

   **macOS / Ubuntu (Linux):**
   ```bash
   python3 -m venv venv 
   source venv/bin/activate
   ```

   **Windows (PowerShell / CMD):**
   ```powershell
   python -m venv venv 
   .\venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Ejecutar el Pipeline de Datos e IA

El pipeline procesa las 6 fuentes de datos geolocalizadas y ajusta el modelo de Fourier y Random Forest. Este comando es igual para todos los sistemas operativos:

```bash
python main.py --run
```
*(Esto generará los archivos consolidados en `internal/processed/` que el dashboard necesita para arrancar).*

---

## 3. Lanzar el Dashboard

El panel principal (Streamlit) se lanza con el siguiente comando (uniforme para todos los sistemas):

```bash
streamlit run dashboard/app.py 
```

En caso de fallo intentar:
```bash
./venv/bin/streamlit run dashboard/app.py 
```
Accede desde tu navegador en `http://localhost:8501`.

---

## 4. Configurar el Motor Inteligente (Ollama + Qwen)

INVICTUS utiliza **Ollama** de forma local para generar reportes narrativos resguardando la privacidad. El modelo por defecto es **Qwen 7B**.

### Requisitos:
- Al menos 8GB RAM.

### 4.1 Instalación de Ollama:

**Ubuntu (Linux):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
- Descarga la aplicación para Mac desde: [ollama.ai/download/mac](https://ollama.ai/download) e instálala en la carpeta Aplicaciones.

**Windows:**
- **Automática:** En una terminal PowerShell ejecuta: `.\setup_ollama.ps1`
- **Manual:** Descarga el ejecutable desde [ollama.ai/download/windows](https://ollama.ai/download) e instálalo.

### 4.2 Descarga del Modelo e Inicio del Servidor:

Una vez instalado Ollama, ejecuta esto en cualquier terminal (Windows/Mac/Linux):

1. **Descarga el modelo:**
   ```bash
   ollama pull qwen:7b
   ```
2. **Inicia el servidor de Ollama** (siempre debe ejecutarse en segundo plano antes de acceder a la pestaña de IA del dashboard):
   ```bash
   ollama serve
   ```
   *El servidor correrá en `http://localhost:11434`.*

---

## 5. Tareas Automáticas (Opcional - Sólo VS Code)

Si utilizas VS Code, el proyecto incluye tareas con un solo clic. Presiona `Ctrl+Shift+K` para acceder a:
- **Iniciar Ollama**
- **Descargar Qwen 7B**
- **Probar Ollama**
- **Iniciar Dashboard**
- **Ejecutar Pipeline**

---

## 6. Solución de Problemas (Troubleshooting)
- **Error de Conexión de Ollama ("Connection refused"):** Ejecuta `ollama serve` en una terminal y déjala abierta.
- **Tarda mucho en generar texto:** Tienes opciones más ligeras. Descarga el modelo `phi:2.7b` con `ollama pull phi:2.7b` y configúralo en el archivo `src/config/ai_constants.py`.
