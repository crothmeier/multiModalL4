# Repository Hygiene Documentation

## CI Guardrails

This repository enforces strict hygiene standards through automated CI workflows and branch protection rules.

### Workflow Status Checks

#### 1. **hygiene** (Required)

Runs all configured pre-commit hooks on:

- Pull requests to `main`
- Direct pushes to `main`

This workflow validates:

- Code formatting (shellcheck, shfmt, ruff, etc.)
- Security scanning (detect-secrets)
- Docker best practices (hadolint)
- YAML validation (yamllint)
- JSON formatting
- File hygiene (trailing whitespace, EOF fixes)

#### 2. **forbid-artifacts** (Required)

Blocks pull requests that introduce forbidden file types:

- Docker Compose backup files matching: `(^|/)docker-compose\.yml\.backup-`
- Model and data artifacts with extensions:
  - `.gguf` (GGML models)
  - `.safetensors` (Stable Diffusion models)
  - `.pt` (PyTorch models)
  - `.onnx` (ONNX models)
  - `.ckpt` (Checkpoint files)
  - `.bin` (Binary model files)
  - `.npz`, `.npy` (NumPy arrays)
  - `.tar`, `.zst`, `.gz`, `.zip` (Archives)
  - `.parquet` (Data files)

### Automated Maintenance

#### Monthly Pre-commit Updates

A scheduled workflow runs on the 1st of each month at 8:00 AM UTC to:

1. Update all pre-commit hooks to latest versions
2. Open a pull request with changes
3. Ensure hooks stay current with upstream fixes

The workflow can also be triggered manually via GitHub Actions UI.

### Branch Protection

#### Enabling Protection

After merging CI guardrails, enable branch protection by running:

```bash
# Ensure scripts are executable
chmod +x scripts/enforce_branch_protection.sh scripts/ci/forbid_artifacts.sh

# Apply protection rules (requires gh CLI with admin access)
./scripts/enforce_branch_protection.sh
```

#### Protection Rules Applied

- **Required status checks**: `hygiene` and `forbid-artifacts` must pass
- **Strict mode**: Branches must be up-to-date before merging
- **Required reviews**: 1 approving review required
- **Dismiss stale reviews**: Reviews dismissed when new commits pushed
- **Code owner reviews**: CODEOWNERS approval required
- **Linear history**: Merge commits disabled
- **Force push protection**: Force pushes blocked
- **Branch deletion protection**: Branch cannot be deleted
- **Admin enforcement**: Rules apply to administrators
- **Conversation resolution**: All PR comments must be resolved

### Bypassing Restrictions

In exceptional cases where artifacts must be added:

1. **Temporary bypass** (NOT RECOMMENDED):

   - Create a dedicated PR with clear justification
   - Requires explicit approval from @crothmeier
   - Must document why the artifact is necessary
   - Consider alternatives (Git LFS, external storage)

2. **Permanent exclusions**:
   - No permanent exclusions are allowed
   - All artifacts should use appropriate storage solutions

### Local Development

#### Running Hooks Locally

Before pushing changes, run:

```bash
pre-commit run --all-files
```

#### Installing Pre-commit

```bash
pip install pre-commit
pre-commit install
```

### Troubleshooting

#### Hook Failures

If pre-commit hooks fail in CI:

1. Pull latest changes
2. Run `pre-commit run --all-files` locally
3. Fix identified issues
4. Commit fixes and push

#### Artifact Detection

If forbidden artifacts are detected:

1. Remove the files from your commit
2. Use `.gitignore` to prevent accidental commits
3. Consider Git LFS for large files that must be versioned

### Monitoring

Check workflow status:

- [Hygiene Workflow](https://github.com/crothmeier/multiModalL4/actions/workflows/hygiene.yml)
- [Forbid Artifacts Workflow](https://github.com/crothmeier/multiModalL4/actions/workflows/forbid-artifacts.yml)
- [Pre-commit Autoupdate](https://github.com/crothmeier/multiModalL4/actions/workflows/pre-commit-autoupdate.yml)
