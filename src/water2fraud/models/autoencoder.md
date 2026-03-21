# <center> **Análisis Arquitectónico: LSTM Hidden State vs. Cell State** </center>

En el diseño de nuestro `LSTMAutoencoder`, la red neuronal utiliza únicamente el estado oculto final (`hidden[-1]`) para generar el vector latente (cuello de botella), descartando el estado de la celda (`cell`). 

A continuación, justificamos matemáticamente esta decisión para nuestro caso de uso (Detección de anomalías en ventanas de 12 meses):

### 1. ¿Qué representan $h_t$ y $c_t$?
* **Hidden State ($h_t$):** Es la "memoria a corto plazo". Contiene la información procesada que la red considera relevante en el paso temporal actual (el mes 12).
* **Cell State ($c_t$):** Es la "memoria a largo plazo" (la cinta transportadora). Su objetivo principal es evitar el desvanecimiento del gradiente (*Vanishing Gradient*) en secuencias muy largas.

### 2. ¿Por qué NO usar el Cell State aquí?
Existen dos formas teóricas de incorporar $c_t$, pero ambas son contraproducentes para nuestro objetivo:
1. **Concatenación (`[hidden, cell]`):** Ensancharía el espacio latente. En un Autoencoder para detección de anomalías, un cuello de botella demasiado permisivo provoca que la red aprenda a reconstruir perfectamente incluso el fraude, aumentando los Falsos Negativos.
2. **Arquitectura Seq2Seq pura:** Pasar la tupla `(hidden, cell)` al Decoder es ideal para tareas de *Forecasting* (predecir el futuro, como en traducción), pero nosotros aplicamos una técnica *RepeatVector* para forzar a la red a "descomprimir" un único concepto global temporal.

### Conclusión
Dado que nuestra longitud de secuencia es extremadamente corta (**12 pasos temporales**), la red LSTM no sufre pérdida de memoria. El estado $h_t$ del mes 12 tiene perfectamente retenidos los patrones de enero. Mantener un cuello de botella estricto basado solo en $h_{12}$ asegura que las viviendas con comportamientos aberrantes no puedan ser comprimidas y, por tanto, su error de reconstrucción se dispare.