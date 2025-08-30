# Repository Hygiene Documentation

## Overview

This document describes the repository hygiene standards and tooling for the multiModalL4 project. These standards ensure consistent code quality, prevent sensitive data leakage, and maintain a clean git history.

## Ignore Patterns Rationale

### Compose Backups
- **Pattern**: `docker-compose.yml.backup-*`
- **Reason**: Automated backup files clutter the repository and create noise in diffs. One curated snapshot is maintained in `migrations/2025-08-compose-backups/curated/` for reference.

### Environment & Secrets
- **Patterns**: `.env`, `*.pem`, `*.key`, `token*`, `hf_*`
- **Reason**: Prevent accidental commit of credentials, API keys, and sensitive configuration.

### Model Artifacts
- **Patterns**: `*.gguf`, `*.safetensors`, `*.pt`, `*.bin`
- **Reason**: Large binary files belong in LFS or external storage, not in git history. These files can be hundreds of MB to several GB.

### Build Artifacts & Caches
- **Patterns**: `__pycache__/`, `.cache/`, `logs/`
- **Reason**: Generated files that change frequently and provide no value in version control.

## Curated Compose Snapshot

The repository maintains ONE curated docker-compose backup in `migrations/2025-08-compose-backups/curated/`. This provides:
- Historical reference for configuration evolution
- Rollback capability if needed
- SHA256 checksums for integrity verification

To update the curated snapshot:
```bash
./scripts/cleanup_compose_backups.sh
```

## Required Commands

### Initial Setup
```bash
# Install pre-commit and hooks
pre-commit autoupdate && pre-commit install

# Or use Makefile
make precommit-bootstrap
```

### Secrets Management
```bash
# Initial baseline creation
detect-secrets scan --baseline .secrets.baseline

# Update baseline after legitimate changes
detect-secrets scan --baseline .secrets.baseline --update .secrets.baseline

# Or use Makefile
make secrets-scan
```

### Running Checks
```bash
# Run all pre-commit hooks
pre-commit run -a

# Or use Makefile
make precommit-run
```

### Repository Cleanup
```bash
# Clean backup files and maintain curated snapshot
make repo-clean
```

## Optional: History Cleanup with git-filter-repo

If the repository history contains numerous backup files that need removal:

### Installation
```bash
# macOS
brew install git-filter-repo

# Linux/WSL
pip install git-filter-repo

# Verify installation
git filter-repo --version
```

### Dry Run (Safe Preview)
```bash
# Clone to test repository
git clone --mirror https://github.com/your-org/multiModalL4.git multiModalL4-test
cd multiModalL4-test

# Analyze what would be removed
git filter-repo --analyze
cat .git/filter-repo/analysis/path-deleted-sizes.txt | grep backup
```

### Actual Purge
```bash
# BACKUP FIRST!
cp -r multiModalL4 multiModalL4-backup

# Remove backup patterns from history
cd multiModalL4
git filter-repo --path-glob 'docker-compose.yml.backup-*' --invert-paths
git filter-repo --path-glob 'migrations/**/docker-compose.yml.backup-*' --invert-paths

# Force push (requires appropriate permissions)
git push origin --force --all
git push origin --force --tags
```

### Collaborator Resync
After history rewrite, all collaborators must:
```bash
# Save any local changes
git stash

# Fetch fresh history
git fetch origin

# Reset to match remote
git reset --hard origin/$(git branch --show-current)

# Reapply local changes if any
git stash pop
```

## Pre-commit Hooks

The repository uses these pre-commit hooks:

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with newline |
| `mixed-line-ending` | Enforce LF line endings |
| `check-merge-conflict` | Prevent committing merge markers |
| `check-yaml/json/toml` | Validate configuration syntax |
| `detect-secrets` | Prevent credential commits |
| `check-added-large-files` | Block files >20MB |
| `shellcheck` | Lint shell scripts |
| `shfmt` | Format shell scripts |
| `hadolint` | Lint Dockerfiles |
| `ruff` | Python linting and formatting |
| `yamllint` | YAML style enforcement |
| `no-commit-to-branch` | Protect main/master branches |

## Troubleshooting

### Pre-commit hook failures
```bash
# See detailed output
pre-commit run --all-files --verbose

# Skip specific hook temporarily
SKIP=hook-name git commit -m "message"
```

### Secrets detected
```bash
# Review detected secrets
detect-secrets audit .secrets.baseline

# Mark false positive in baseline and update
detect-secrets scan --baseline .secrets.baseline --update .secrets.baseline
```

### Large file blocked
```bash
# Check file size
du -h path/to/file

# Add to LFS if appropriate
git lfs track "*.extension"
git add .gitattributes
git add path/to/file
```

## Maintenance Schedule

- **Weekly**: Run `make repo-clean` to maintain hygiene
- **Before PR**: Run `make precommit-run` to catch issues
- **Monthly**: Review and update `.secrets.baseline`
- **Quarterly**: Review `.gitignore` patterns for optimization
