"""M1: generación del espacio de diseño.

Etapa 1: operadores enumerativos deterministas sobre las semillas (variaciones
cercanas, auditables). Etapa 2: mutación abierta con SAFE / REINVENT4 sobre los
mejores de la etapa 1. Ambas pasan por `filters.hard_filter` y por el filtro de
síntesis (SAScore < 4.5, RAScore > 0.7).
"""
