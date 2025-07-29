#!/usr/bin/env bats

@test "fetch_models.sh exists and is executable" {
    [ -x scripts/fetch_models.sh ]
}

@test "fetch_models.sh has proper shell settings" {
    grep -q "set -euo pipefail" scripts/fetch_models.sh
}

@test "fetch_models.sh defines all required URLs" {
    grep -q "MISTRAL_URL=" scripts/fetch_models.sh
    grep -q "DEEPSEEK_CODER_URL=" scripts/fetch_models.sh
    grep -q "DEEPSEEK_VL_URL=" scripts/fetch_models.sh
}