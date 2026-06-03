#!/usr/bin/env bash
# Attach this Coder workspace to the Langfuse docker network and start socat
# forwarders so localhost:3005/9090 reach the sibling service containers (the
# Coder proxy forwards this workspace's localhost). Run via `make remote`.
# See memory: coder-docker-sibling-networking.
set -u

NETWORK="brave-rag-langfuse_default"
MAPPINGS=(
  "3005 langfuse-web 3000"   # Langfuse web UI
  "9090 minio 9000"          # MinIO (S3) for media uploads
)

# Our own container id (from the bind-mounts in /proc/self/mountinfo).
SELF="$(grep '/containers/' /proc/self/mountinfo | grep -oE '[0-9a-f]{64}' | head -1)"

# Attach to the Langfuse network — silently no-ops if already attached.
docker network connect "$NETWORK" "$SELF" 2>/dev/null

# Start one socat forwarder per mapping, skipping ports already forwarded.
for spec in "${MAPPINGS[@]}"; do
  read -r lp th tp <<<"$spec"
  if ! pgrep -f "TCP-LISTEN:$lp," >/dev/null; then
    socat "TCP-LISTEN:$lp,fork,reuseaddr" "TCP:$th:$tp" &
    echo "forwarding localhost:$lp -> $th:$tp"
  fi
done
