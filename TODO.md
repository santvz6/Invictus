# Gestión Inteligente de la Huella Hídrica Turística: Detección de Consumos Anómalos y Reutilización Predictiva de Agua.

## 0. Preparación y Alineación del Equipo
- [x] **Acceso y Permisos:** Todos los integrantes tienen acceso al repositorio de GitHub y permisos de escritura.
- [x] **Workflow de Git:** Establecer reglas de ramas (ej. `main` para entregas, `dev` para integración, y ramas por tarea para cada perfil).
- [x] **Entorno de Desarrollo (Setup):** 
    - [x] Creación de un entorno virtual común (`venv` o `conda`).
    - [x] Archivo `requirements.txt` con librerías base: `pandas`, `numpy`, `networkx`, `scikit-learn` y `matplotlib`.
- [ ] **Sincronización Interdisciplinar:**
    - [ ] **Físicos:** Definir las leyes de conservación o variables de contorno (aislamiento de Tabarca).
    - [ ] **Informáticos:** Estructurar la arquitectura del Grafo (Nodos = Barrios, Aristas = Proximidad/Similitud).


## Análisis y Exploración de Datos (EDA) 
- [ ] Comprobar la regularidad en el comportamiento del *Ratio de Consumo* de forma aislada para cada barrio. 
- [ ] Si el comportamiento es regular, tratar cada barrio como un dato (correspondiente a la media). *De esta forma podremos tratar con 57 filas y detectar comportamientos inusuales respecto a otros barrios*.
- [ ] Comparar barrios con el barrio de **TABARCA**.


## Modelado
- [ ] Convertir nuestros datos en un Grafo.