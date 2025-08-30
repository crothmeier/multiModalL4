#!/usr/bin/env bash
set -euo pipefail

# Cleanup compose backups and maintain curated snapshot
# Usage: ./scripts/cleanup_compose_backups.sh

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly CURATED_DIR="${REPO_ROOT}/migrations/2025-08-compose-backups/curated"
readonly BACKUP_PATTERN="docker-compose.yml.backup-*"

echo "=== Docker Compose Backup Cleanup ==="
cd "${REPO_ROOT}"

# Step 1: Unstage any backup files from git index
echo "→ Unstaging backup files from git index..."
git_unstage_count=0
while IFS= read -r -d '' file; do
  if git ls-files --cached --others --exclude-standard | grep -q "^${file}$"; then
    git reset HEAD "${file}" 2> /dev/null || true
    ((git_unstage_count++))
  fi
done < <(find . -name "${BACKUP_PATTERN}" -print0)

if [[ ${git_unstage_count} -gt 0 ]]; then
  echo "  Unstaged ${git_unstage_count} backup file(s)"
else
  echo "  No backup files in git index"
fi

# Step 2: Find most recent backup for curation
echo "→ Finding most recent backup for curation..."
latest_backup=""
latest_timestamp=0

while IFS= read -r -d '' file; do
  # Extract timestamp from filename (format: backup-YYYYMMDD-HHMMSS)
  basename_file=$(basename "${file}")
  if [[ "${basename_file}" =~ backup-([0-9]{8}-[0-9]{6})$ ]]; then
    timestamp="${BASH_REMATCH[1]}"
    # Convert to comparable format
    timestamp_num="${timestamp//-/}"
    if [[ ${timestamp_num} -gt ${latest_timestamp} ]]; then
      latest_timestamp=${timestamp_num}
      latest_backup="${file}"
    fi
  fi
done < <(find . -name "${BACKUP_PATTERN}" -type f -print0)

# Step 3: Optionally move latest backup to curated directory
if [[ -n "${latest_backup}" ]]; then
  echo "  Found latest backup: ${latest_backup}"

  # Create curated directory if needed
  if [[ ! -d "${CURATED_DIR}" ]]; then
    echo "→ Creating curated directory..."
    mkdir -p "${CURATED_DIR}"
  fi

  # Copy to curated (preserve name)
  curated_file="${CURATED_DIR}/$(basename "${latest_backup}")"
  if [[ ! -f "${curated_file}" ]]; then
    echo "→ Preserving latest backup in curated directory..."
    cp "${latest_backup}" "${curated_file}"
    echo "  Copied to: ${curated_file}"

    # Generate SHA256SUMS for curated files
    echo "→ Generating SHA256SUMS for curated files..."
    (cd "${CURATED_DIR}" && sha256sum ./*.backup-* > SHA256SUMS 2> /dev/null || true)
  else
    echo "  Latest backup already curated"
  fi
else
  echo "  No backup files found"
fi

# Step 4: Verify gitignore is working
echo "→ Verifying .gitignore patterns..."
test_file="docker-compose.yml.backup-test"
touch "${test_file}"

if git check-ignore "${test_file}" > /dev/null 2>&1; then
  echo "  ✓ Root backup pattern correctly ignored"
else
  echo "  ✗ WARNING: Root backup pattern not ignored!"
  rm -f "${test_file}"
  exit 1
fi
rm -f "${test_file}"

# Test migrations subdirectory pattern
test_migration_file="migrations/test/docker-compose.yml.backup-test"
mkdir -p "$(dirname "${test_migration_file}")"
touch "${test_migration_file}"

if git check-ignore "${test_migration_file}" > /dev/null 2>&1; then
  echo "  ✓ Migration backup pattern correctly ignored"
else
  echo "  ✗ WARNING: Migration backup pattern not ignored!"
  rm -f "${test_migration_file}"
  rmdir "$(dirname "${test_migration_file}")" 2> /dev/null || true
  exit 1
fi
rm -f "${test_migration_file}"
rmdir "$(dirname "${test_migration_file}")" 2> /dev/null || true

# Step 5: Final check - ensure no backups would be added
echo "→ Final verification..."
untracked_backups=$(git ls-files --others --exclude-standard | grep -c "${BACKUP_PATTERN}" || true)
if [[ ${untracked_backups} -gt 0 ]]; then
  echo "  ✓ Found ${untracked_backups} backup file(s) properly ignored"
fi

staged_backups=$(git diff --cached --name-only | grep -c "${BACKUP_PATTERN}" || true)
if [[ ${staged_backups} -gt 0 ]]; then
  echo "  ✗ ERROR: ${staged_backups} backup file(s) still staged!"
  echo "  Run: git reset HEAD \$(git diff --cached --name-only | grep '${BACKUP_PATTERN}')"
  exit 1
fi

echo ""
echo "✓ Cleanup complete!"
echo "  - Backup files unstaged from git index"
echo "  - Latest backup preserved in curated directory (if any)"
echo "  - .gitignore patterns verified working"
