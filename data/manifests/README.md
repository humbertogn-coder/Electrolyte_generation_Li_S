# data/manifests

Un CSV por campaña. Git guarda el índice, Grace guarda los bytes.

Columnas obligatorias:

```
campaign_id, record_id, record_type, smiles, config_hash, tier, grace_path, sha256, created_at, commit
```

- `record_id`: `PSM-xxxxxx` (molécula) o `PACES-xxxx` (formulación).
- `record_type`: `tier0`, `tier1_md`, `tier2_dft`, `tier3_aimd`, `export_kmc`.
- `config_hash`: SHA-256 del YAML de campaña resuelto (con herencia aplicada).
- `grace_path`: ruta absoluta en `$SCRATCH` de Grace al directorio del cálculo.
- `sha256`: checksum del archivo principal de salida (trayectoria, `.out` de ORCA, `OUTCAR`).

Nunca se suben trayectorias ni funciones de onda a este repositorio.
