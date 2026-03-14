"""
Mapeo ponderado de Barrios AMAEM → Municipios INE.

Estructura: {barrio: [(codigo_ine, peso), ...]}
  - codigo_ine: string con el código del municipio (sin nombre), ej. '03014'
  - peso: float en [0.0, 1.0] que indica qué fracción del barrio pertenece a ese municipio.
  - Los pesos de un barrio suman <= 1.0 (la fracción restante se considera sin cobertura/NaN).

"""

# Pesos que suman 0 → el barrio no tiene datos en el INE (NaN)
_SIN_DATOS: list = []

BARRIO_MUNICIPIO_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    # ── Barrios 100% núcleo urbano de Alicante ─────────────────────────────────
    "1-BENALUA":                     [("03014 Alacant/Alicante", 1.00)],
    "2-SAN ANTON":                   [("03014 Alacant/Alicante", 1.00)],
    "3-CENTRO":                      [("03014 Alacant/Alicante", 1.00)],
    "4-MERCADO":                     [("03014 Alacant/Alicante", 1.00)],
    "5-CAMPOAMOR":                   [("03014 Alacant/Alicante", 1.00)],
    "6-LOS ANGELES":                 [("03014 Alacant/Alicante", 1.00)],
    "7-SAN AGUSTIN":                 [("03014 Alacant/Alicante", 1.00)],
    "8-ALIPARK":                     [("03014 Alacant/Alicante", 1.00)],
    "9-FLORIDA ALTA":                [("03014 Alacant/Alicante", 1.00)],
    "10-FLORIDA BAJA":               [("03014 Alacant/Alicante", 1.00)],
    "11-CIUDAD DE ASIS":             [("03014 Alacant/Alicante", 1.00)],
    "12-POLIGONO BABEL":             [("03014 Alacant/Alicante", 1.00)],
    "13-SAN GABRIEL":                [("03014 Alacant/Alicante", 1.00)],
    "14-ENSANCHE DIPUTACION":        [("03014 Alacant/Alicante", 1.00)],
    "15-POLIGONO SAN BLAS":          [("03014 Alacant/Alicante", 1.00)],
    "16-PLA DEL BON REPOS":          [("03014 Alacant/Alicante", 1.00)],
    "17-CAROLINAS ALTAS":            [("03014 Alacant/Alicante", 1.00)],
    "18-CAROLINAS BAJAS":            [("03014 Alacant/Alicante", 1.00)],
    "19-GARBINET":                   [("03014 Alacant/Alicante", 1.00)],
    "22-CASCO ANTIGUO - SANTA CRUZ": [("03014 Alacant/Alicante", 1.00)],
    "23-RAVAL ROIG -V. DEL SOCORRO":[("03014 Alacant/Alicante", 1.00)],
    "24-SAN BLAS - SANTO DOMINGO":   [("03014 Alacant/Alicante", 1.00)],
    "25-ALTOZANO - CONDE LUMIARES":  [("03014 Alacant/Alicante", 1.00)],
    "26-SIDI IFNI - NOU ALACANT":    [("03014 Alacant/Alicante", 1.00)],
    "27-SAN FERNANDO-PRIN. MERCEDES":[("03014 Alacant/Alicante", 1.00)],
    "28-EL PALMERAL":                [("03014 Alacant/Alicante", 1.00)],
    "29-URBANOVA":                   [("03014 Alacant/Alicante", 1.00)],
    "30-DIVINA PASTORA":             [("03014 Alacant/Alicante", 1.00)],
    "31-CIUDAD JARDIN":              [("03014 Alacant/Alicante", 1.00)],
    "32-VIRGEN DEL REMEDIO":         [("03014 Alacant/Alicante", 1.00)],
    "33- MORANT -SAN NICOLAS BARI":  [("03014 Alacant/Alicante", 1.00)],
    "34-COLONIA REQUENA":            [("03014 Alacant/Alicante", 1.00)],
    "35-VIRGEN DEL CARMEN":          [("03014 Alacant/Alicante", 1.00)],
    "36-CUATROCIENTAS VIVIENDAS":    [("03014 Alacant/Alicante", 1.00)],
    "37-JUAN XXIII":                 [("03014 Alacant/Alicante", 1.00)],
    "40-CABO DE LAS HUERTAS":        [("03014 Alacant/Alicante", 1.00)],
    "55-PUERTO":                     [("03014 Alacant/Alicante", 1.00)],

    # ── Barrios con influencia/cercanía a otros municipios (CPs y Maps) ───────
    
    # Norte: Conexión con San Vicente del Raspeig
    "20-RABASA":                     [("03014 Alacant/Alicante", 0.60), ("03122 Sant Vicent del Raspeig/San Vicente del Raspeig", 0.40)],
    "21-TOMBOLA":                    [("03014 Alacant/Alicante", 0.70), ("03122 Sant Vicent del Raspeig/San Vicente del Raspeig", 0.30)],

    # Noreste: Conexión con Sant Joan d'Alacant (CPs 03550 / 03540)
    "38-VISTAHERMOSA":               [("03014 Alacant/Alicante", 0.70), ("03119 Sant Joan d'Alacant", 0.30)],
    "39-ALBUFERETA":                 [("03014 Alacant/Alicante", 0.80), ("03119 Sant Joan d'Alacant", 0.20)],
    "41-PLAYA DE SAN JUAN":          [("03014 Alacant/Alicante", 0.60), ("03119 Sant Joan d'Alacant", 0.40)],

    # ── Zonas Industriales (Frontera Sur con Elche) ────────────────────────────
    "53-POLIGONO ATALAYAS":          [("03014 Alacant/Alicante", 0.90), ("03065 Elx/Elche", 0.10)],
    "54-POLIGONO VALLONGA":          [("03014 Alacant/Alicante", 0.80), ("03065 Elx/Elche", 0.20)], # Corregido: Alcalalí no tenía sentido
    "56-DISPERSOS":                  [("03014 Alacant/Alicante", 1.00)],

    # ── Pedanías / Núcleos periféricos distribuidos por cercanía y CP ──────────
    
    # Frontera Sur / Suroeste (Elche)
    "BACAROT":                       [("03014 Alacant/Alicante", 0.80), ("03065 Elx/Elche", 0.20)],
    "FONTCALENT":                    [("03014 Alacant/Alicante", 0.80), ("03065 Elx/Elche", 0.20)],
    "PDA VALLONGA":                  [("03014 Alacant/Alicante", 0.80), ("03065 Elx/Elche", 0.20)],
    "REBOLLEDO":                     [("03014 Alacant/Alicante", 0.80), ("03065 Elx/Elche", 0.20)],

    # Frontera Oeste (Agost)
    "LA ALCORAYA":                   [("03014 Alacant/Alicante", 0.60), ("03002 Agost", 0.40)], # CP 03698 compartido con Agost

    # Frontera Noroeste (San Vicente del Raspeig - Usan el CP 03690 de Sant Vicent)
    "LA CAÑADA":                     [("03122 Sant Vicent del Raspeig/San Vicente del Raspeig", 0.70), ("03014 Alacant/Alicante", 0.30)],
    "MORALET":                       [("03122 Sant Vicent del Raspeig/San Vicente del Raspeig", 0.70), ("03014 Alacant/Alicante", 0.30)],
    "VERDEGAS":                      [("03122 Sant Vicent del Raspeig/San Vicente del Raspeig", 0.70), ("03014 Alacant/Alicante", 0.30)],

    # Frontera Norte (Mutxamel - Usa el CP 03110)
    "MONNEGRE":                      [("03090 Mutxamel", 0.70), ("03014 Alacant/Alicante", 0.30)],
    "VILLAFRANQUEZA":                [("03014 Alacant/Alicante", 0.60), ("03122 Sant Vicent del Raspeig/San Vicente del Raspeig", 0.20), ("03090 Mutxamel", 0.20)],

    # Zona Noreste (Mixto con Sant Joan)
    "SANTA FAZ":                     [("03014 Alacant/Alicante", 0.50), ("03119 Sant Joan d'Alacant", 0.50)],

    # Marítimo (Logística marítima desde Santa Pola)
    "TABARCA":                       [("03121 Santa Pola", 0.70), ("03014 Alacant/Alicante", 0.30)],
}