# 🤖 Integración del LLM (Ollama + Qwen) en INVICTUS

Este documento explica la arquitectura, archivos y justificación técnica detrás de la integración de Inteligencia Artificial Generativa en el proyecto. 

---

## 🎯 Por Qué Ollama
Debido a la naturaleza de los datos utilizados (datos demográficos, facturación de agua) era crucial un enfoque centrado en la **Privacidad de los Datos (100% Local)**, con el fin de evitar que los datos viajen a APIs en la nube. Ollama nos permite incrustar modelos de código abierto sin esfuerzo de manera rápida, asegurando que cumple con los requerimientos necesarios.

---

## 📦 Archivos Involucrados y Arquitectura

### 1. `src/utils/ollama_client.py` 
Es el cliente Python especializado para comunicarse con el servidor local Ollama.
- ✅ *Health Check*: Verifica disponibilidad antes de efectuar llamadas.
- ✅ *Manejo de Errores*: Intercepta fallos de LLM de manera fluida.
- ✅ *Text Generation*: Soporta generación simple, streaming e incorporación de contexto.

### 2. `dashboard/components/llm_report.py`
El panel visual en la Tab 3 de Streamlit:
- Implementa *fallbacks* elegantes: Si Ollama no está operativo, puede mostrar reportes pre-generados o avisos, no bloqueando la ejecución del panel principal.
- Detecta dinámicamente si el motor responde y permite el re-análisis según la ciudad o barrio escogido.

### 3. `src/config/ai_constants.py` y `.env`
Las variables del modelo se definen globalmente:
```python
LLM_MODEL = "qwen:7b"
LLM_BASE_URL = "http://localhost:11434"
LLM_TIMEOUT = 120
LLM_TEMPERATURE = 0.7
```

### 4. Automatización de entorno
- **`.vscode/tasks.json`**: Simplifica comandos rutinarios del entorno para el usuario.
- **`setup_ollama.ps1`**: Asegura una instalación y *pull* del modelo Qwen 7B de manera transparente, especialmente útil en entornos de Windows.

---

## 📊 Diagrama de Ingesta y Generación

```text
┌────────────────────────────────────────────────────────────┐
│                  DASHBOARD STREAMLIT (UI)                  │
│  ✨ Acción: Usuario solicita "Informe de Anomalías"         │
└────────────────────────────────────────────────────────────┘
                            ↓                           
┌────────────────────────────────────────────────────────────┐
│                OLLAMA CLIENT (ollama_client.py)            │
│  Formatea datos físicos + Z-Score para la generación       │
└────────────────────────────────────────────────────────────┘
                            ↓ (HTTP POST sobre 11434)
┌────────────────────────────────────────────────────────────┐
│         OLLAMA SERVER LOCAL (Model: qwen:7b en Memoria)    │
│  Genera conclusiones explicativas en base a los modelos    │
└────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Modelos Soportados y Alternativas
El sistema viene preparado por defecto con **Qwen 7B** por ser un balance excepcional entre calidad analítica y costo de RAM (~4.7GB). 
Si se cuenta con recursos muy limitados u hardware de servidores diferente, pueden modificarse fácilmente:

- `phi:2.7b` (ideal para latencia muy baja, ocupa 1.6GB)
- `mistral:latest` (ocupa ~4.1GB, con gran capacidad de razonamiento)
- `neural-chat:7b` (optimizado para chatbots conversacionales)

Luego, en `src/config/ai_constants.py`, se cambiaría el nombre de acuerdo al nuevo modelo. El ecosistema deja abierto agregar flujos adicionales fácilmente.
