# Water2Fraud - Detección de fraudes y anomalías

## Introducción
Seguramente estemos ante un problema con datos sin etiquetar (aprendizaje no supervisado). Tendríamos que utilizar datos como:
- Consumos brutos
- Tipos de contrato (doméstico/industrial)
- Ubicación
- Tipo de Vivienda (dimensiones u otros datos).


## Pipeline
### 1. Segmentación y Perfilado (Unsupervised)
Este es el primer motor. Su función es agrupar todos los contadores de la ciudad sin saber qué son, basándose puramente en su "huella dactilar" de consumo (picos, horas, fines de semana).

- **Algoritmos**: K-Means, DBSCAN
- **Output:** ID de Cluster (Ej: Cluster 0, Cluster 1, Cluster 2).

***Nota:** Aquí la IA todavía no sabe qué es cada grupo.*


### 2. Modelado matemático/físico
Analizamos los centroides (el comportamiento promedio) de cada clusert y aplicamos nuestras reglas:

- p.ej.: Si un grupo tiene un consumo pico los fines de semana y casi cero de lunes a jueves, eso es físicamente una vivienda turística.

Definimos cada cluster con una etiqueta (doméstico, turístico/comercial, desocupada, posible fuga).

### 3. Detección de Discrepancia y Confianza
Compara la etiqueta anterior con el contrato real de Aguas de Alicante.

- **Output**: lista de nuestras viviendas sospechosas.


### 4. Detección de Fraude/Anomalías - Confianza
Dentro de nuestros sopechosos detectados el algoritmo calcula  un **Anomaly Score.**

- **Algoritmos**: Isolation Forest o Local Outlier Factor (LOF)
- **Output:** Ranking de viviendas para inspeccionar (prioridad en base al Anomaly Score).

