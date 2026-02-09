#!/bin/bash

PGVECTOR_VERSION="0.8.1-pg18-trixie"
PG_TEXTSEARCH_VERSION="0.5.0-dev" # requires >=0.5.0 due to text_config parsing issues

DATE_TAG="260130"
IMAGE_TAG="${DATE_TAG}-${PGVECTOR_VERSION}-${PG_TEXTSEARCH_VERSION}"

POSTGRES_PORT=9010

# VOLUME_NAME="Qwen3-Embedding-0.6B"
# VOLUME_NAME="voyage-4-nano"
VOLUME_NAME="kanana-nano-2.1b-embedding"
VOLUME_NAME="KURE-v1"

# mount: https://velog.io/@dailylifecoding/docker-postgres-after-version-18-volume-error
docker run -it --rm \
    --env-file ./.env \
    --shm-size=32g \
    -p ${POSTGRES_PORT:-5432}:5432 \
    -v ./data/${VOLUME_NAME}:/var/lib/postgresql/18/docker \
    id4thomas/pgvec-textsearch-ko:${IMAGE_TAG}

# -v ${DOCKER_VOLUME_NAME}:/var/lib/postgresql \