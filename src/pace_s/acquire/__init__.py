"""M3 (b): adquisición multiobjetivo y gestión del bucle de active learning.

Adquisición: qNEHVI (BoTorch). Diferencia deliberada frente al Expected
Improvement de objetivo único de Nat. Commun. 2025; evita pesos arbitrarios.
Lotes de 60 formulaciones, 3 iteraciones. Iteración 1 sembrada por diversidad
y por ComBat. La comparación contra selección aleatoria es obligatoria (H4).
"""
