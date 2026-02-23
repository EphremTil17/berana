#!/bin/bash
# -----------------------------------------------------------------------------
# Berana-Trans: Label Studio Environment Setup
# -----------------------------------------------------------------------------
# This script specifically initializes the Docker environment for Label Studio.
# It is decoupled from the main pipeline setup since Label Studio is primarily
# a training-data annotation tool, not required for production inference.
# -----------------------------------------------------------------------------
set -e

# Dynamically calculate project paths
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# --- TERMINAL COLORS ---
BLUE='\033[0;34m'
SUCCESS='\033[0;32m'
RESET='\033[0m'

echo -e "${BLUE}============================================================${RESET}"
echo -e "${BLUE} Berana-Trans: Label Studio Docker Initialization ${RESET}"
echo -e "${BLUE}============================================================${RESET}"

# Label Studio Pre-flight
# Pre-create the data directory with host-user ownership so the container
# (which runs as host UID via docker-compose 'user:' directive) can write to it.
# This prevents the '[Errno 13] Permission denied' crash loop on first run.
echo -e "${BLUE}[1/2] Preparing Label Studio data volume...${RESET}"
LS_DATA_DIR="$PROJECT_ROOT/tools/label_studio/data"
LS_ENV_FILE="$PROJECT_ROOT/tools/label_studio/.env"
LS_DATA_ENV="$LS_DATA_DIR/.env"
LS_ENV_TEMPLATE="$PROJECT_ROOT/tools/label_studio/.env.example"
LS_DATA_ENV_TEMPLATE="$LS_DATA_DIR/.env.example"
mkdir -p "$LS_DATA_DIR"

if [ ! -f "$LS_ENV_FILE" ]; then
    cp "$LS_ENV_TEMPLATE" "$LS_ENV_FILE"
fi
if [ ! -f "$LS_DATA_ENV" ]; then
    cp "$LS_DATA_ENV_TEMPLATE" "$LS_DATA_ENV"
fi

# Inject actual host UID/GID into the Label Studio .env so docker-compose
# runs the container process as the correct user.
HOST_UID=$(id -u)
HOST_GID=$(id -g)
sed -i "s/^HOST_UID=.*/HOST_UID=${HOST_UID}/" "$LS_ENV_FILE"
sed -i "s/^HOST_GID=.*/HOST_GID=${HOST_GID}/" "$LS_ENV_FILE"

# CRITICAL: Label Studio reads its OWN internal data/.env at Django startup
# via its custom entrypoint. docker-compose 'environment:' and 'env_file:'
# are both ignored. We must write these vars directly to the data volume .env.
# The SECRET_KEY line is preserved if it already exists from a prior run.
touch "$LS_DATA_ENV"
for VAR in \
    "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true" \
    "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/berana_data"; do
    KEY="${VAR%%=*}"
    if grep -q "^${KEY}=" "$LS_DATA_ENV" 2>/dev/null; then
        sed -i "s|^${KEY}=.*|${VAR}|" "$LS_DATA_ENV"
    else
        echo "$VAR" >> "$LS_DATA_ENV"
    fi
done

# Populate SECRET_KEY for Label Studio internal startup env if it is still placeholder.
if grep -q "^SECRET_KEY=replace_with_local_secret_key" "$LS_DATA_ENV"; then
    GENERATED_SECRET_KEY=$(head -c 32 /dev/urandom | base64 | tr -d '\n')
    sed -i "s|^SECRET_KEY=.*|SECRET_KEY=${GENERATED_SECRET_KEY}|" "$LS_DATA_ENV"
fi

echo -e "${BLUE}[2/2] Finalizing...${RESET}"
echo -e "${SUCCESS}  Label Studio ready. Container will run as UID ${HOST_UID}:${HOST_GID}${RESET}"

echo -e "${BLUE}============================================================${RESET}"
echo -e "${SUCCESS} Label Studio Setup Completed Successfully.${RESET}"
echo -e " To start Label Studio:  ${SUCCESS}cd tools/label_studio && docker compose up -d${RESET}"
echo -e " Label Studio dashboard:  http://localhost:8080"
echo -e "${BLUE}============================================================${RESET}"
