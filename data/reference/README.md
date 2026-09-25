# data/reference

Datos de referencia y calibración. Todo lo que está aquí es pequeño, versionable y con fuente.

| Archivo | Contenido | Estado |
|---|---|---|
| `seeds.csv` | Semillas coordinantes y diluyentes (M0). | EMP sin SMILES; F5DEE y OFE marcados para confirmar contra PubChem |
| `salts.csv` | LiFSI y LiTFSI. | listo |
| `tier0_reference_molecules.csv` | Cinco moléculas con orden esperado de tier 0 (criterio de éxito de H1). | falta rellenar DN numérico con fuente |
| `joule2021_table1_solubility.csv` | Tabla 1 de *Joule* 2021: solubilidad de S₈ y Li₂Sₓ en mM para 40+ sistemas. **Conjunto de la compuerta H3.** | **por digitalizar (semana 0)** |
| `combat_seed.csv` | Subconjunto de ComBat / MISPR (Atwi y Rajput) para arranque en frío de los sustitutos. | por descargar |
| `scirep2023_69solvents.csv` | 69 solventes con ΔmixG de COSMO-RS (*Sci. Rep.* 2023). | solicitar a los autores |

Columnas mínimas de `joule2021_table1_solubility.csv`:

```
system_id, solvent, diluent, salt, salt_conc_M, category, donor_number, species, solubility_mM_S, temperature_C, source_note
```

`species` toma valores `S8`, `Li2S8`, `Li2S6`, `Li2S4`, `Li2S2`, `Li2S`. `category` toma `moderately`, `sparingly`, `highly` (solvatante), tal como organiza la revisión.

Los datos de CE de literatura, si se compilan, van en un archivo aparte con las condiciones (carga de S, E/S, densidad de corriente, ventana de voltaje, paper) como columnas obligatorias. Solo se usan como validación cualitativa de orden, dentro de un mismo paper.
