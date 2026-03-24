- [ ] Actualmente nuestro AE-LSTM funciona como un detector de anomalías general (perfil esperado). Es decir, no tiene en cuenta solamente el valor esperado de consumo. Por tanto, habrá que implementar una ponderación para el **error__consumo** y el **error__viviendas_ilegales_%** que representa una tercera gráfica. 

- [ ] El menu lateral desplegable del dashboard sólo tiene en cuenta las anomalías del Autoencoder. Podríamos poner un sistema de filtros para mostrar:
    - [ ] AE anomalías
    - [ ] Físicos anomalías
    - [ ] los FRAUD_RISK_SCORE anomalías en orden descendente

- [ ] Mejorar la optimización de hiperparámetros para el AE.