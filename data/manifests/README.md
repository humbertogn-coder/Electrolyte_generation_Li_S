# data/manifests

One CSV per campaign. Git keeps the index, Grace keeps the bytes.

Mandatory columns:

```
campaign_id, record_id, record_type, smiles, config_hash, tier, grace_path, sha256, created_at, commit
```

- `record_id`: `PSM-xxxxxx` (molecule) or `PACES-xxxx` (formulation).
- `record_type`: `tier0`, `tier1_md`, `tier2_dft`, `tier3_aimd`, `export_kmc`.
- `config_hash`: SHA-256 of the resolved campaign YAML (inheritance applied).
- `grace_path`: absolute path in Grace's `$SCRATCH` to the calculation directory.
- `sha256`: checksum of the main output file (trajectory, ORCA `.out`, `OUTCAR`).

Trajectories and wavefunctions are never committed to this repository.
