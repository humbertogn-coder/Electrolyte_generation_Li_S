# PACE-S

**P**ipeline de generación y cribado **AC**tivo de **E**lectrolitos para baterías de Li metal con azufre (Li-S, con transferencia a Li-SPAN).

Balbuena Group, Texas A&M University. Repositorio independiente del modelo kMC 3D de morfología (`kmc3d`): PACE-S puede depender de `kmc3d`, `kmc3d` nunca depende de PACE-S.

Posicionamiento en una frase: existe el mapa de los solventes conocidos para Li-S; falta la máquina que proponga los desconocidos y que llegue hasta morfología.

## Qué hace

```
M0  Semillas, espacio de diseño y datos de calibración
M1  Generador por fragmentos y mutaciones      -->  20k-60k SMILES
M2  Tier 0: filtros + descriptores baratos     -->  ~3k supervivientes
M3  Modelos sustitutos + adquisición qNEHVI    <--+
M4  Tier 1: MLIP-MD (oráculo principal)        ---+  bucle de active learning
    Tier 2: DFT (finalistas)
    Tier 3: AIMD (top 3, mecanismo)
M5  SHAP + exportación de parámetros           -->  kMC 3D de morfología
```

## Estructura

```
contracts/          esquemas JSON congelados (semana 0) y ejemplos
src/pace_s/
  generate/         semillas, operadores de mutación, filtros duros y de síntesis
  descriptors/      tier 0: xtb, CREST, morfeus, RDKit
  oracles/          tier 1 MLIP-MD, tier 2 DFT, tier 3 AIMD
  surrogate/        modelos sustitutos con incertidumbre
  acquire/          qNEHVI (BoTorch), gestión del bucle
  analysis/         SHAP, Pareto, figuras
  export/           formato propio + adaptador a kmc3d
workflows/
  slurm/            plantillas de job array (CPU) y GPU para Grace
  configs/          YAML por campaña; los pesos Li-S vs Li-SPAN viven aquí
data/
  manifests/        índices y checksums de cada campaña (NO trayectorias)
  reference/        Tabla 1 de Joule 2021, ComBat, moléculas de referencia
tests/
notebooks/
```

## Instalación

```bash
conda env create -f environment.yml          # generación, tier 0, sustitutos, AL
conda activate pace-s
pip install -e .
pytest
```

Los stacks de MLIP (MACE, fairchem/UMA, DeePMD) y LAMMPS van en un entorno separado (`environment-mlip.yml`) porque sus dependencias de PyTorch/CUDA chocan con el resto. En Grace, LAMMPS con plugin de MACE o DeePMD se compila aparte; pregunta por la ruta que ya resolvió el grupo antes de compilar desde cero.

## Reglas de trabajo

1. **Manifiestos versionados.** Un CSV por campaña en `data/manifests/` con id, SMILES, hash de configuración, ruta en Grace y checksum. Git guarda el índice, Grace guarda los bytes.
2. **`campaign_id` en cada salida.** Sin esto, tres iteraciones de bucle son irreconstruibles.
3. **Procedencia en cada número.** Tier y nivel de teoría, siempre. Ningún valor de tier 0 entra al paper.
4. **Tests desde el primer commit.** SMILES válidos, JSON que valida contra su esquema, moléculas de referencia con valores esperados en tier 0.
5. **Limitaciones escritas dentro del código.** Nernst-Einstein sobreestima la conductividad; los MLIP no son reactivos; PBE-D3 sobreestima tasas de descomposición hasta 9 órdenes de magnitud (los potenciales de reducción se calculan con funcional híbrido).

## Contratos congelados

- `contracts/kmc_interface.schema.json`: lo que PACE-S entrega al kMC 3D. Los tres campos de `kinetics` son la entrada obligatoria del kMC.
- `contracts/candidate_table.schema.json`: una fila de la tabla maestra de candidatos.

Cambiar un contrato requiere subir `schema_version` y actualizar el adaptador `src/pace_s/export/kmc3d_adapter.py`, que está fijado a una versión concreta de `kmc3d`.

## Cambiar de sistema

Li-S y Li-SPAN comparten código. Difieren en `workflows/configs/campaign_LiS.yaml` y `campaign_LiSPAN.yaml`: pesos de objetivos (O2 solubilidad de polisulfuros domina en Li-S; O3 pasivación reductiva domina en Li-SPAN).

## Documentos asociados

Propuesta v2 (`propuesta_pipeline_electrolitos_LiS.md`) y revisión de literatura v2 (`revision_literatura_generacion_electrolitos.md`), en la carpeta padre del proyecto.

## Licencia

Por definir antes de hacer público el repositorio.
