#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# multimodal-llm Vault bootstrap (non-interactive)
# - Ubuntu 24.04, Vault 1.13
# - Zero prompts: script exits with error if required data is absent
###############################################################################

# ---------- 1. Configuration discovery --------------------------------------
VAULT_ADDR="${VAULT_ADDR:-https://vault.homelab.local:8200}"
VAULT_NAMESPACE="${VAULT_NAMESPACE:-}"

# Hugging Face token: env → ~/.huggingface/token → exit
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(grep -m1 -E '^[a-zA-Z0-9_-]{20,}' "$HOME/.huggingface/token" 2> /dev/null || true)"
fi
[[ -z "${HF_TOKEN:-}" ]] && {
  echo "❌ HF_TOKEN not found (set env var or ~/.huggingface/token)"
  exit 1
}

# MinIO credentials: env → ~/.mc/config.json → random
if [[ -z "${MINIO_ACCESS_KEY:-}" || -z "${MINIO_SECRET_KEY:-}" ]]; then
  if [[ -f "$HOME/.mc/config.json" ]]; then
    read -r MINIO_ACCESS_KEY MINIO_SECRET_KEY < \
      <(jq -r '.aliases.models | .accessKey+" "+.secretKey' "$HOME/.mc/config.json" 2> /dev/null | head -n1)
  fi
fi
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minio$(openssl rand -hex 8)}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-$(openssl rand -hex 32)}"

# ---------- 2. JWT key-pair --------------------------------------------------
KEY_DIR="/etc/vault/multimodal-llm"
mkdir -p "$KEY_DIR"
JWT_PRIV="$KEY_DIR/jwt-private.pem"
JWT_PUB="$KEY_DIR/jwt-public.pem"

if [[ ! -s "$JWT_PRIV" || ! -s "$JWT_PUB" ]]; then
  openssl genrsa -out "$JWT_PRIV" 2048
  openssl rsa -in "$JWT_PRIV" -pubout -out "$JWT_PUB"
  chmod 600 "$JWT_PRIV" "$JWT_PUB"
fi

JWT_SECRET="$(openssl rand -base64 32)"
JWT_ALGORITHM="RS256"

echo "→ Configuring Vault objects for multimodal-llm..."
export VAULT_ADDR VAULT_NAMESPACE

# ---------- 3. Vault provisioning -------------------------------------------
# Enable AppRole auth method
vault auth enable -path=multimodal-llm approle 2> /dev/null || true

# Create policy
vault policy write multimodal-llm - << EOF
path "secret/data/multimodal-llm" { capabilities = ["read"] }
path "auth/token/renew-self" { capabilities = ["update"] }
EOF

# Create role
vault write auth/multimodal-llm/role/llm-agent \
  token_policies=multimodal-llm \
  token_ttl=1h token_max_ttl=24h \
  secret_id_ttl=0 secret_id_num_uses=0

# Store secrets
vault kv put secret/multimodal-llm \
  HF_TOKEN="$HF_TOKEN" \
  JWT_SECRET="$JWT_SECRET" \
  JWT_ALGORITHM="$JWT_ALGORITHM" \
  JWT_PRIVATE_KEY=@"$JWT_PRIV" \
  JWT_PUBLIC_KEY=@"$JWT_PUB" \
  MINIO_ACCESS_KEY="$MINIO_ACCESS_KEY" \
  MINIO_SECRET_KEY="$MINIO_SECRET_KEY"

# Get credentials
ROLE_ID=$(vault read -field=role_id auth/multimodal-llm/role/llm-agent/role-id)
SECRET_ID=$(vault write -f -field=secret_id auth/multimodal-llm/role/llm-agent/secret-id)

# Save credentials
echo "$ROLE_ID" > "$KEY_DIR/role-id"
echo "$SECRET_ID" > "$KEY_DIR/secret-id"
chmod 600 "$KEY_DIR"/role-id "$KEY_DIR"/secret-id

echo "✅ Vault bootstrap complete"
echo "   Role ID: $KEY_DIR/role-id"
echo "   Secret ID: $KEY_DIR/secret-id"
echo "   JWT keys: $JWT_PRIV, $JWT_PUB"
