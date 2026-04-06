<!-- OLLAMA_SETUP.md -->

# 🤖 Configuración de Ollama + Qwen para INVICTUS

Este proyecto está configurado para usar **Ollama** con el modelo **Qwen 7B** para generar reportes inteligentes en el dashboard.

Ollama te permite ejecutar modelos LLM localmente sin enviar datos a servidores externos, respetando la confidencialidad de los datos de telelectura.

---

## ✅ Requisitos Previos

- **Windows 10/11** con acceso administrativo
- **8+ GB de RAM RAM** (recomendado 16GB)
- **Ollama** instalado: https://ollama.ai

---

## 🚀 Instalación Rápida

### 1️⃣ Descargar e Instalar Ollama

1. Ve a https://ollama.ai
2. Descarga el instalador para Windows
3. Ejecuta el instalador y sigue las instrucciones
4. Ollama se instalará en `C:\Users\{tu_usuario}\AppData\Local\Programs\Ollama`

**Verificar instalación:**
```powershell
ollama --version
```

### 2️⃣ Descargar el modelo Qwen 7B

En una terminal PowerShell (puede tardar 5-10 minutos según tu velocidad):

```powershell
ollama pull qwen:7b
```

**O usa la tarea desde VS Code:**
- Abre la Paleta de Comandos: `Ctrl+Shift+K`
- Busca: "Tasks: Run Task"
- Selecciona: "📥 Descargar modelo Qwen 7B"

### 3️⃣ Iniciar el servidor Ollama

En una terminal PowerShell:

```powershell
ollama serve
```

Deberías ver algo como:
```
2026/04/06 00:45:00 "Listening on 127.0.0.1:11434"
```

**O usa la tarea desde VS Code:**
- Abre la Paleta de Comandos: `Ctrl+Shift+K`
- Busca: "Tasks: Run Task"
- Selecciona: "🚀 Iniciar Ollama (Qwen)"

---

## 🧪 Verificar que funciona

### Opción 1: Probar desde PowerShell

```powershell
curl -X POST http://localhost:11434/api/generate -d '{\"model\":\"qwen:7b\",\"prompt\":\"Hola\",\"stream\":false}' | Select-Object -ExpandProperty Content | ConvertFrom-Json
```

O usa la tarea desde VS Code:
- Abre la Paleta de Comandos: `Ctrl+Shift+K`
- Busca: "Tasks: Run Task"
- Selecciona: "🧪 Probar Ollama"

### Opción 2: Usar desde Python

```python
from src.utils.ollama_client import OllamaLLM

llm = OllamaLLM(model="qwen:7b")
print(llm.generate("¿Qué es el agua?"))
```

---

## 📊 Usar en el Dashboard

Una vez que Ollama esté corriendo:

1. **Inicia el servidor Ollama** en una terminal:
   ```powershell
   ollama serve
   ```

2. **Inicia el Dashboard** en otra terminal:
   ```powershell
   streamlit run dashboard/app.py
   ```

3. En el dashboard, selecciona un barrio y haz clic en **"✨ Generar Informe"**

El modelo Qwen analizará los datos del barrio y generará un reporte automáticamente.

---

## 📋 Referencia de Tareas VS Code

Presiona `Ctrl+Shift+K` para abrir el ejecutor de tareas:

| Tarea | Descripción |
|-------|-------------|
| 🚀 Iniciar Ollama | Lanza el servidor en background |
| 📥 Descargar Qwen 7B | Descarga el modelo (~4.7GB) |
| 🧪 Probar Ollama | Verifica la conexión |
| 📊 Iniciar Dashboard | Lanza Streamlit |
| ▶️ Ejecutar Pipeline | Corre el procesamiento de datos |

---

## 🛠️ Troubleshooting

### Error: "Connection refused"
- **Solución**: Asegúrate que Ollama está corriendo con `ollama serve`

### Error: "Model not found"
- **Solución**: Descarga el modelo: `ollama pull qwen:7b`

### La generación es muy lenta
- **Causa**: CPU débil o insuficiente RAM
- **Solución**: Usa un modelo más pequeño: `ollama pull qwen:4b`

### Ollama no se inicia
- **Solución**: Reinicia la máquina o reinstala Ollama

---

## 📚 Modelos alternativos

Si Qwen 7B es lento, prueba otros modelos:

```powershell
# Modelos pequeños (más rápidos)
ollama pull phi:2.7b          # ~1.6GB, muy rápido
ollama pull neural-chat:7b    # ~4.1GB

# Modelos medianos
ollama pull mistral:latest    # ~4.1GB
ollama pull llama2:13b        # ~7.4GB

# Modelos grandes (más precisos)
ollama pull neural-chat:34b   # ~20GB
```

Luego, en `src/utils/ollama_client.py` cambia:
```python
llm = OllamaLLM(model="tu-modelo-aqui:version")
```

---

## 📖 Más información

- **Ollama Docs**: https://github.com/ollama/ollama
- **Qwen Docs**: https://github.com/QwenLM/Qwen
- **API Ollama**: http://localhost:11434/api/tags

---

¿Preguntas? Consulta el código en `src/utils/ollama_client.py` 🚀
