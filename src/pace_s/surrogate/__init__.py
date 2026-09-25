"""M3 (a): modelos sustitutos con incertidumbre.

Representación: descriptores tier 0 + fingerprints de Morgan reducidos +
variables de formulación. Modelos: ensemble de gradient boosting con
incertidumbre por cuantiles, más proceso gaussiano para la adquisición.
Chemprop como referencia. Arranque en frío sembrado con ComBat.

Cada predicción se reporta con `distance_to_seeds` (1 - Tanimoto máximo frente
a las semillas): el generador extrapola, y hay que decir cuánto.
"""
