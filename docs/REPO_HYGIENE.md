# Repository Hygiene (Multimodal)

- **No weights/binaries in Git.** Only configs, code, and small text assets.
- CI enforces:
  - `hygiene`: pre-commit hooks
  - `forbid-artifacts`: blocks model/backups/binaries
- Prefer PVs or bootstrap Jobs to seed models for Triton/LLaVA/Whisper.
