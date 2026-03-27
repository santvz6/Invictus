# Invictus: Master Guide para Auditoría, Testing y Escalado Arquitectónico

## 🎯 Objetivo de la Misión
Este documento sirve como el protocolo maestro para la evolución del proyecto **Invictus**. Tu misión es realizar una auditoría exhaustiva, implementar una cobertura de tests robusta y refactorizar la relación entre el núcleo lógico (`src/`) y la visualización (`dashboard/`) para asegurar un sistema 100% sincronizado y escalable.

**REGLA DE ORO:** La precisión y la búsqueda minuciosa de errores son prioritarias sobre la velocidad de entrega. No des por hecho que el código actual es infalible.

---

## 📂 Arquitectura y Contexto Técnico
El proyecto utiliza un stack basado en:
*   **Core**: Python 3.x
*   **Procesamiento**: Pandas, NumPy, Scikit-learn.
*   **IA/ML**: Modelos LSTM Autoencoder para detección de anomalías.
*   **Dashboard**: Streamlit.
*   **Configuración**: Centralizada en `src/config/`.

### Estructura Clave a Auditar:
1.  **`src/config/`**: Gestiona `paths.py` y `string_keys.py`. Estos deben ser la única fuente de verdad para rutas y nombres de columnas.
2.  **`src/water2fraud/features/`**: Contiene los procesadores de datos (AMAEM, INE, GVA, etc.) y el orquestador `preprocessor.py`.
3.  **`src/water2fraud/models/`**: Lógica de entrenamiento e inferencia de los modelos de IA.
4.  **`dashboard/`**: Interfaz de usuario que carga datos a través de `data_loader.py`.

---

## 🛠 PARTE 1: Auditoría Lógica y Verificación Funcional
Debes actuar como un ingeniero de QA Senior.

1.  **Auditoría de Código (Archivo por Archivo)**:
    *   Revisa cada script buscando inconsistencias en el flujo de datos.
    *   Verifica que no haya "hardcoding" de valores que deberían estar en `src/config/`.
    *   Analiza la lógica de cálculo de scores (ej. `AE_SCORE_WEIGHTED`) para asegurar coherencia matemática.
2.  **Verificación en Tiempo de Ejecución**:
    *   Ejecuta el pipeline desde la terminal (ej. `python main.py`).
    *   Verifica que la generación de archivos en `internal/data/` sea correcta.
    *   Levanta el dashboard y comprueba que todas las métricas visualizadas coincidan con los datos procesados.
    *   Detecta cualquier `RuntimeWarning` o errores silenciosos en el procesamiento de señales (ej. `NaN`s inesperados).

---

## 🧪 PARTE 2: Framework de Unit Testing
Desarrolla una base de tests que proteja la integridad de la lógica.

1.  **Implementación con `pytest`**:
    *   Crea una estructura de tests paralela en el directorio `tests/`.
    *   Diseña tests para los procesadores individuales en `src/water2fraud/features/`.
2.  **Casos de Prueba Críticos**:
    *   Validación de entradas vacías o corruptas.
    *   Persistencia de la forma de los tensores generados por el preprocesador.
    *   Inversión correcta de escalado (log1p -> expm1).
3.  **Flexibilidad de Flujo**: Puedes implementar los tests como herramienta de diagnóstico durante la Parte 1 si lo consideras más eficiente para encontrar errores ocultos.

---

## 🚀 PARTE 3: Sincronización Arquitectónica y Escalabilidad
Este es el pilar de la escalabilidad del proyecto. Actualmente, existe el riesgo de que el dashboard implemente lógica propia que no está en el core.

1.  **Eliminación de Lógica Duplicada**:
    *   Audita `dashboard/data_loader.py`. Si encuentras cálculos de scores, preprocesamiento o "parches" de seguridad, muévelos a la clase correspondiente en `src/water2fraud/`.
    *   El dashboard debe ser una capa de **solo lectura y visualización**.
2.  **Dependencia Directa**:
    *   Asegura que el dashboard importe y utilice las funciones de `src/` para cualquier cálculo dinámico.
    *   Si realizo una mejora en `src/water2fraud/features/preprocessor.py`, el dashboard debe beneficiarse de ella sin tocar una sola línea de código en `dashboard/`.
3.  **Principios de Diseño**:
    *   Aplica **SOLID** donde sea posible para facilitar la adición de nuevos procesadores o modelos sin romper los existentes.
    *   Asegura que el sistema sea capaz de manejar nuevos barrios o fuentes de datos solo actualizando la configuración.

---

## 📝 Checkpoint Final
Antes de dar por terminada la tarea, pregúntate:
*   ¿Si cambio el algoritmo de cálculo de riesgo en `src/`, el dashboard se actualiza solo?
*   ¿He probado casos extremos que podrían romper la lógica de negocio?
*   ¿El código es legible y profesional para otros desarrolladores?

**Ejecuta, prueba, audita y construye de forma implacable.**
