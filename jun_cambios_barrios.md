# Integración Geográfica y Nuevos Datos Externos

Este documento resume los cambios implementados para enriquecer el dataset de consumos de agua de AMAEM a nivel de **Barrio** con fuentes externas de turismo y movilidad.

## 1. Contexto del Problema
Previamente, en la exploración inicial, el consumo de agua de AMAEM (información altamente precisa a nivel local) se cruzó con datos crudos del INE sobre **Plazas Turísticas**. Esto presentaba tres problemas principales:
1. **Falta de granularidad y alcance:** Solo evaluaba estimaciones estáticas de Viviendas Turísticas (VUT), y dejaba fuera a los hoteles.
2. **Ausencia de información temporal:** Los datos del INE usados no fluctuaban mes a mes, perdiéndose la evolución de aperturas y cierres de establecimientos.
3. **Mide capacidad, no turismo real:** Las plazas son camas disponibles, pero esto no equivale a cuántas personas realmente han visitado la zona, lo cual es el motor real que empuja el consumo de agua.

## 2. Solución: Las 3 Nuevas Fuentes de Datos Procesadas
Para solventar esto, localizamos datos crudos de la Generalitat Valenciana (GVA) y del Instituto Nacional de Estadística (INE), y los hemos limipiado y agregado en **3 nuevos archivos CSV**.

### A. `gva_municipios_vt.csv` (Viviendas Turísticas, GVA)
- **Origen:** Archivo `m-viviendas-2022-2025.csv` (Generalitat Valenciana).
- **Tratamiento:** El archivo original contiene el registro individual de miles de VUTs de toda la Comunidad, con sus fechas exactas de alta y baja administrativa. Se programó un script (`prepare_gva.py`) que reconstruye mes a mes el stock vivo de plazas turísticas para L'Alacantí.
- **Valor aportado:** Introduce la variable dinámica de la oferta extrahotelera. Ahora se sabe exactamente si en "Junio de 2023" el municipio disponía de 500 plazas de alquiler más que el mes anterior.

### B. `gva_municipios_hoteles.csv` (Hoteles, GVA)
- **Origen:** Archivo `m-hoteles-2022-2026.csv` (Generalitat Valenciana).
- **Tratamiento:** Se iteró sobre cada hotel (por registro individual) leyendo su fecha de inicio y baja de actividad.
- **Valor aportado:** Complementa la oferta de alojamiento. El consumo de agua hotelero (spa, piscinas, lavandería) tiene un impacto estructural fuertísimo en la red hídrica, que el primer modelo omitía por completo.

### C. `ine_tmov_municipios.csv` (Turistas Reales por Telefonía Móvil - INE)
- **Origen:** Excels crudos de 2019 a 2025 (`exp_tmov_emisor_mun_*.xlsx`).
- **Tratamiento:** Usando `prepare_ine_tmov.py`, iteramos mes a mes, aislando solo los municipios objetivo, sumando los turistas (extrayéndolos de una estructura Excel compleja con múltiples encabezados).
- **Valor aportado:** Este es el **termómetro del turismo real**. Mide flujos poblacionales gracias al rastro de los teléfonos móviles. Si en un mes se registran 60.000 móviles de foráneos adicionales (ej: fiestas locales), esto se correlacionará fuertemente con los picos anómalos de consumo que registre AMAEM.

## 3. Integración Final de Vuelta a los Barrios (`jun_exploracion.ipynb`)

Estos tres CSVs producen una serie temporal continua `(Mes, Municipio, Valor)`. Sin embargo, el consumo de agua lo estudiamos a nivel local `(Mes, Barrio)`.

Para tender el puente entre ambos mundos, hemos reesctructurado el archivo **`barrio_mapping.py`**. En él:
1. Aseguramos que los barrios limítrofes, fronterizos o las pedanías apunten administrativamente a su municipio correcto (ejemplo corregido: *Vallonga* antes apuntaba a Alcalalí, ahora a Alicante).
2. Repartimos la afluencia turística municipal (viajeros del INE o camas de la GVA) hacia cada distrito/barrio AMAEM de su jurisdicción, en función de su peso relacional.

El último paso realizado fue inyectar **3 celdas ejecutable al final del cuaderno `jun_exploracion.ipynb`**. Estas celdas toman nuestros CSVs recién salidos del horno, ejecutan combinaciones matriciales usando nuestro mapa probabilístico `barrio_mapping.py` corregido, y terminan fundiendo las 3 nuevas columnas sobre el *DataFrame* final (`df_final`). 

De esta forma, en una sola tabla, los algoritmos predictivos disponen de: *consumo mensual por barrio, sus temperaturas, su densidad, sus plazas locales de hotelería activas y una inferencia matemática precisa del impacto del turismo real (móviles)*.
