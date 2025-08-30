#!/bin/sh
# POSIX-compliant auto-patch script for docker-compose.yml

set -e

# Check for yq
if ! command -v yq > /dev/null 2>&1; then
  echo "Error: yq is required but not installed" >&2
  exit 1
fi

# Check yq version
YQ_VERSION=$(yq --version | sed -n 's/.*version v\([0-9.]*\).*/\1/p')
YQ_MAJOR=$(echo "$YQ_VERSION" | cut -d. -f1)
YQ_MINOR=$(echo "$YQ_VERSION" | cut -d. -f2)
if [ "$YQ_MAJOR" -lt 4 ] || { [ "$YQ_MAJOR" -eq 4 ] && [ "$YQ_MINOR" -lt 40 ]; }; then
  echo "Error: yq version 4.40 or higher required (found $YQ_VERSION)" >&2
  exit 1
fi

COMPOSE_FILE="docker-compose.yml"
BACKUP_FILE="docker-compose.yml.backup-$(date +%Y%m%d-%H%M%S)"
CHANGES_MADE=0

# Backup if not already backed up this session
if [ ! -f "$BACKUP_FILE" ]; then
  cp "$COMPOSE_FILE" "$BACKUP_FILE"
fi

echo "Checking docker-compose.yml compliance..."

# Patch 1 - HF_TOKEN must be required
# Use a temp file approach for complex updates
TEMP_FILE=$(mktemp)
trap 'rm -f "$TEMP_FILE"' EXIT

# Check current HF_TOKEN value
if yq eval '.["x-vllm-base"].environment' "$COMPOSE_FILE" | grep -q '^- HF_TOKEN='; then
  current_hf=$(yq eval '.["x-vllm-base"].environment' "$COMPOSE_FILE" | grep '^- HF_TOKEN=' | sed 's/^- //')
else
  current_hf=""
fi

need_hf="HF_TOKEN=\${HF_TOKEN:?Error: HF_TOKEN environment variable is required}"
if [ "$current_hf" != "$need_hf" ]; then
  echo "Updating HF_TOKEN requirement..."
  # Update HF_TOKEN in the environment array
  yq eval '."x-vllm-base".environment |= map(select(test("^HF_TOKEN=") | not))' -i "$COMPOSE_FILE"
  yq eval ".\"x-vllm-base\".environment += [\"$need_hf\"]" -i "$COMPOSE_FILE"
  CHANGES_MADE=1
fi

# Patch 2 - GPU headroom for LLaVA service
# Find service with --served-model-name=llava
llava_service=""
for service in $(yq eval '.services | keys | .[]' "$COMPOSE_FILE"); do
  if yq eval ".services.\"$service\".command[]" "$COMPOSE_FILE" 2> /dev/null | grep -q '^--served-model-name=llava$'; then
    llava_service="$service"
    break
  fi
done

if [ -n "$llava_service" ]; then
  # Check current GPU utilization
  if yq eval ".services.\"$llava_service\".command" "$COMPOSE_FILE" | grep -q '^- --gpu-memory-utilization='; then
    current_gpu=$(yq eval ".services.\"$llava_service\".command" "$COMPOSE_FILE" | grep '^- --gpu-memory-utilization=' | sed 's/^- //')
  else
    current_gpu=""
  fi

  target_gpu='--gpu-memory-utilization=0.10'
  if [ "$current_gpu" != "$target_gpu" ]; then
    echo "Setting GPU memory utilization to 0.10 for $llava_service..."
    # Replace the gpu-memory-utilization line in the command array
    yq eval ".services.\"$llava_service\".command = (.services.\"$llava_service\".command | map(if test(\"^--gpu-memory-utilization=\") then \"$target_gpu\" else . end))" "$COMPOSE_FILE" > "$TEMP_FILE"
    mv "$TEMP_FILE" "$COMPOSE_FILE"
    CHANGES_MADE=1
  fi
fi

echo
if [ $CHANGES_MADE -eq 0 ]; then
  echo "✓ docker-compose.yml already compliant"
else
  echo "✓ docker-compose.yml updated successfully"
  echo "  Backup saved to: $BACKUP_FILE"
fi

exit 0
