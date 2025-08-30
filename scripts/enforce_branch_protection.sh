#!/usr/bin/env bash
set -euo pipefail

# Branch protection enforcement script for GitHub
# Usage: ./scripts/enforce_branch_protection.sh [repo] [branch]

REPO="${1:-crothmeier/multiModalL4}"
BRANCH="${2:-main}"

echo "Enforcing branch protection on ${REPO}@${BRANCH}"
echo "================================================"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check authentication
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

# Apply branch protection rules
echo "Applying protection rules..."
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO}/branches/${BRANCH}/protection" \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=hygiene" \
  -f "required_status_checks[contexts][]=forbid-artifacts" \
  -f "enforce_admins=true" \
  -f "required_pull_request_reviews[required_approving_review_count]=1" \
  -f "required_pull_request_reviews[dismiss_stale_reviews]=true" \
  -f "required_pull_request_reviews[require_code_owner_reviews]=true" \
  -f "required_linear_history=true" \
  -f "allow_force_pushes=false" \
  -f "allow_deletions=false" \
  -f "required_conversation_resolution=true" \
  -f "lock_branch=false" \
  -f "allow_fork_syncing=false"

# Check if the command succeeded
if gh api "/repos/${REPO}/branches/${BRANCH}/protection" >/dev/null 2>&1; then
    echo ""
    echo "✅ Branch protection successfully applied!"
    echo ""
    echo "Protected branch: ${BRANCH}"
    echo "Repository: ${REPO}"
    echo ""
    echo "Rules enforced:"
    echo "  - Required status checks: hygiene, forbid-artifacts (strict)"
    echo "  - Require 1 approving review"
    echo "  - Dismiss stale reviews on new commits"
    echo "  - Require code owner reviews"
    echo "  - Enforce linear history"
    echo "  - Enforce for administrators"
    echo "  - Block force pushes and deletions"
    echo "  - Require conversation resolution"
    echo ""
    echo "Verify settings at: https://github.com/${REPO}/settings/branches"
else
    echo ""
    echo "❌ Failed to apply branch protection."
    echo "Ensure you have admin permissions on ${REPO}"
    exit 1
fi
