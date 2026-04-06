"""
ollama_client.py
================
Cliente para integración con Ollama (LLM local).
Permite usar modelos como Qwen sin llamadas a APIs externas.

Uso:
    llm = OllamaLLM(model="qwen:7b")
    respuesta = llm.generate("Tu pregunta aquí")
"""

import requests
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Cliente para comunicarse con Ollama (servidor LLM local).
    
    Requiere:
    - Ollama instalado: https://ollama.ai
    - Modelo descargado: ollama pull qwen:7b
    - Servidor corriendo: ollama serve
    """
    
    def __init__(self, 
                 model: str = "qwen:7b", 
                 base_url: str = "http://localhost:11434",
                 timeout: int = 120):
        """
        Inicializa el cliente Ollama.
        
        Args:
            model: Nombre del modelo en Ollama (default: qwen:7b)
            base_url: URL del servidor Ollama (default: localhost:11434)
            timeout: Timeout en segundos para las peticiones (default: 120s)
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        
    def health_check(self) -> bool:
        """Verifica si Ollama está disponible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Ollama no disponible: {e}")
            return False
    
    def list_models(self) -> list[str]:
        """Lista los modelos disponibles en Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Error listando modelos: {e}")
            return []
    
    def generate(self, 
                 prompt: str, 
                 stream: bool = False,
                 temperature: float = 0.7,
                 top_k: int = 40,
                 top_p: float = 0.9) -> str:
        """
        Genera texto usando Ollama.
        
        Args:
            prompt: Texto de entrada
            stream: Si True, devuelve streaming (más rápido para UI)
            temperature: Creatividad (0-1, menor = más determinista)
            top_k: Tokens a considerar en cada paso
            top_p: Nucleus sampling (0-1)
            
        Returns:
            Texto generado por el modelo
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            
            if stream:
                result = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        result += data.get("response", "")
                return result
            else:
                data = response.json()
                return data.get("response", "")
                
        except requests.exceptions.ConnectionError:
            error_msg = (
                f"❌ No se puede conectar con Ollama en {self.base_url}\n"
                "Asegúrate de que:\n"
                "1. Ollama está instalado (https://ollama.ai)\n"
                "2. El servidor está corriendo (ejecuta: ollama serve)\n"
                "3. El modelo está descargado (ejecuta: ollama pull qwen:7b)"
            )
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            logger.error(f"Error generando texto: {e}")
            return f"Error: {str(e)}"
    
    def generate_with_context(self,
                             prompt: str,
                             context: Optional[str] = None,
                             **kwargs) -> str:
        """
        Genera texto con contexto adicional (útil para análisis de datos).
        
        Args:
            prompt: Pregunta principal
            context: Contexto relevante (datos, métricas, etc.)
            **kwargs: Otros parámetros para generate()
            
        Returns:
            Respuesta del modelo
        """
        if context:
            prompt_with_context = (
                f"Contexto:\n{context}\n\n"
                f"Pregunta:\n{prompt}"
            )
        else:
            prompt_with_context = prompt
            
        return self.generate(prompt_with_context, **kwargs)
