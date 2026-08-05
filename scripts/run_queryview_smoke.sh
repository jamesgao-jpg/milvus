#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/deployments/docker/dev/docker-compose-apple-silicon.yml"
MILVUS_BIN="$REPO_ROOT/bin/milvus"
SMOKE_TEST="$REPO_ROOT/tests/python_client/queryview_smoke.py"
VENV_DIR="${QUERYVIEW_SMOKE_VENV:-$REPO_ROOT/.venv-smoke}"
M1A_TEST_PLAN_DIR="${QUERYVIEW_M1A_TEST_PLAN_DIR:-}"
M1A_PYTHON="${QUERYVIEW_M1A_PYTHON:-}"
M1A_MEMORY_MODE="${QUERYVIEW_M1A_MEMORY_MODE:-}"
M1A_MEMORY_CONCURRENCY="${QUERYVIEW_M1A_MEMORY_CONCURRENCY:-32}"
M1A_MEMORY_TOP_K="${QUERYVIEW_M1A_MEMORY_TOP_K:-16384}"
M1A_SHARDS_NUM="${QUERYVIEW_M1A_SHARDS_NUM:-2}"
DISABLE_ITERATOR_STREAMING="${QUERYVIEW_DISABLE_ITERATOR_STREAMING:-false}"

RUN_ID="${QUERYVIEW_SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_ROOT="${QUERYVIEW_SMOKE_RUN_ROOT:-$REPO_ROOT/_artifacts/queryview-smoke}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
INFRA_DIR="$RUN_DIR/infra"
LOCAL_DIR="$RUN_DIR/local"
COMPOSE_OVERRIDE_FILE="$RUN_DIR/docker-compose.override.yml"
COMPOSE_PROJECT_NAME="qv-smoke-${RUN_ID,,}"
CLUSTER_ID="qvsmoke${RUN_ID//[^[:alnum:]]/}"

MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
MILVUS_PORT="${MILVUS_PORT:-19532}"
MINIO_PORT="${QUERYVIEW_SMOKE_MINIO_PORT:-19000}"
MINIO_CONSOLE_PORT="${QUERYVIEW_SMOKE_MINIO_CONSOLE_PORT:-19001}"
METRICS_BASE="${QUERYVIEW_SMOKE_METRICS_BASE:-19090}"

declare -a MILVUS_PIDS=()
declare -a MILVUS_ROLES=()

log() {
    printf '[queryview-smoke] %s\n' "$*"
}

fail() {
    log "ERROR: $*"
    return 1
}

compose() {
    COMPOSE_PROJECT_NAME="$COMPOSE_PROJECT_NAME" \
        DOCKER_VOLUME_DIRECTORY="$INFRA_DIR" \
        docker compose \
        -f "$COMPOSE_FILE" \
        -f "$COMPOSE_OVERRIDE_FILE" \
        "$@"
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    set +e

    for ((i = ${#MILVUS_PIDS[@]} - 1; i >= 0; i--)); do
        if kill -0 "${MILVUS_PIDS[i]}" 2>/dev/null; then
            log "Stopping ${MILVUS_ROLES[i]} (process group ${MILVUS_PIDS[i]})"
            kill -TERM -- "-${MILVUS_PIDS[i]}" 2>/dev/null
        fi
    done

    for pid in "${MILVUS_PIDS[@]}"; do
        wait "$pid" 2>/dev/null
    done

    compose down --remove-orphans >/dev/null 2>&1

    if ((status != 0)); then
        log "Smoke test failed. Logs are retained at $RUN_DIR"
        for log_file in "$LOG_DIR"/*.log; do
            [[ -e "$log_file" ]] || continue
            printf '\n===== %s (last 60 lines) =====\n' "$(basename "$log_file")"
            tail -n 60 "$log_file"
        done
    else
        log "Smoke test passed. Logs are retained at $RUN_DIR"
    fi

    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_tcp() {
    local name=$1
    local host=$2
    local port=$3
    local timeout_seconds=$4
    local deadline=$((SECONDS + timeout_seconds))

    until (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; do
        if ((SECONDS >= deadline)); then
            fail "$name did not listen on $host:$port within ${timeout_seconds}s"
            return 1
        fi
        sleep 1
    done
}

wait_for_http() {
    local name=$1
    local url=$2
    local timeout_seconds=$3
    local deadline=$((SECONDS + timeout_seconds))

    until curl --fail --silent --show-error "$url" >/dev/null 2>&1; do
        if ((SECONDS >= deadline)); then
            fail "$name did not become healthy at $url within ${timeout_seconds}s"
            return 1
        fi
        sleep 1
    done
}

wait_for_role() {
    local role=$1
    local pid=$2
    local metrics_port=$3
    local timeout_seconds=$4
    local deadline=$((SECONDS + timeout_seconds))
    local health_url="http://127.0.0.1:${metrics_port}/healthz"

    until curl --fail --silent --show-error "$health_url" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            fail "$role exited before becoming healthy; see $LOG_DIR/$role.log"
            return 1
        fi
        if ((SECONDS >= deadline)); then
            fail "$role did not become healthy at $health_url within ${timeout_seconds}s"
            return 1
        fi
        sleep 1
    done
}

start_role() {
    local role=$1
    local metrics_port=$2
    local role_local_dir="$LOCAL_DIR/$role"

    mkdir -p "$role_local_dir"
    log "Starting $role (metrics port $metrics_port)"

    (
        export ETCD_ENDPOINTS="127.0.0.1:2379"
        export MINIO_ADDRESS="127.0.0.1:$MINIO_PORT"
        export PULSAR_ADDRESS="pulsar://127.0.0.1:6650"
        export MQ_TYPE="pulsar"
        export ETCD_ROOTPATH="$CLUSTER_ID"
        export MINIO_ROOTPATH="$CLUSTER_ID"
        export MSGCHANNEL_CHANNAMEPREFIX_CLUSTER="$CLUSTER_ID"
        export LOCALSTORAGE_PATH="$role_local_dir"
        export METRICS_PORT="$metrics_port"
        export PROXY_PORT="$MILVUS_PORT"
        export PROXY_QUERYVIEW_DISABLEITERATORSTREAMING="$DISABLE_ITERATOR_STREAMING"
        export LD_LIBRARY_PATH="$REPO_ROOT/internal/core/output/lib:${LD_LIBRARY_PATH:-}"

        if [[ -f "$REPO_ROOT/internal/core/output/lib/libjemalloc.so" ]]; then
            export LD_PRELOAD="$REPO_ROOT/internal/core/output/lib/libjemalloc.so"
            export MALLOC_CONF="background_thread:true"
        fi

        cd "$REPO_ROOT"
        exec setsid "$MILVUS_BIN" run "$role" --run-with-subprocess
    ) >"$LOG_DIR/$role.log" 2>&1 &

    local pid=$!
    MILVUS_PIDS+=("$pid")
    MILVUS_ROLES+=("$role")
    wait_for_role "$role" "$pid" "$metrics_port" 240
}

ensure_venv() {
    if [[ ! -x "$VENV_DIR/bin/python" ]] \
        || ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        log "Creating Python virtual environment at $VENV_DIR"
        python3 -m venv --clear "$VENV_DIR"
    fi

    if ! "$VENV_DIR/bin/python" -c \
        'import pymilvus; assert pymilvus.__version__ == "3.1.0rc69"' \
        >/dev/null 2>&1; then
        log "Installing pymilvus==3.1.0rc69"
        "$VENV_DIR/bin/python" -m pip install \
            --find-links https://test.pypi.org/simple/pymilvus/ \
            pymilvus==3.1.0rc69
    fi
}

for command in docker curl python3 setsid; do
    command -v "$command" >/dev/null || fail "required command is missing: $command"
done

docker compose version >/dev/null
[[ -x "$MILVUS_BIN" ]] || fail "Milvus binary is missing or not executable: $MILVUS_BIN"
[[ -f "$COMPOSE_FILE" ]] || fail "Compose file is missing: $COMPOSE_FILE"
[[ -f "$SMOKE_TEST" ]] || fail "Smoke test is missing: $SMOKE_TEST"
[[ "$DISABLE_ITERATOR_STREAMING" == "true" || "$DISABLE_ITERATOR_STREAMING" == "false" ]] \
    || fail "QUERYVIEW_DISABLE_ITERATOR_STREAMING must be true or false"
if [[ -n "$M1A_TEST_PLAN_DIR" ]]; then
    [[ -x "$M1A_PYTHON" ]] || fail "M1A Python is missing or not executable: $M1A_PYTHON"
    [[ -f "$M1A_TEST_PLAN_DIR/scripts/prepare_openai_50k.py" ]] \
        || fail "M1A preparation script is missing"
    [[ -f "$M1A_TEST_PLAN_DIR/scripts/run_m1a_openai_correctness.py" ]] \
        || fail "M1A correctness script is missing"
    if [[ -n "$M1A_MEMORY_MODE" ]]; then
        [[ "$M1A_MEMORY_MODE" == "batch" || "$M1A_MEMORY_MODE" == "iterator" ]] \
            || fail "M1A memory mode must be batch or iterator"
        [[ -f "$M1A_TEST_PLAN_DIR/scripts/run_m1a_proxy_memory.py" ]] \
            || fail "M1A Proxy memory script is missing"
    fi
elif [[ -n "$M1A_MEMORY_MODE" ]]; then
    fail "M1A memory mode requires QUERYVIEW_M1A_TEST_PLAN_DIR"
fi

mkdir -p "$LOG_DIR" "$INFRA_DIR" "$LOCAL_DIR"

cat >"$COMPOSE_OVERRIDE_FILE" <<EOF
services:
  minio:
    ports: !override
      - "127.0.0.1:${MINIO_PORT}:9000"
      - "127.0.0.1:${MINIO_CONSOLE_PORT}:9001"
EOF

log "Run ID: $RUN_ID"
log "Disable iterator streaming: $DISABLE_ITERATOR_STREAMING"
log "Starting etcd, Pulsar, and MinIO"
compose up -d etcd pulsar minio

wait_for_http "etcd" "http://127.0.0.1:2379/health" 180
wait_for_tcp "Pulsar" "127.0.0.1" 6650 240
wait_for_http "MinIO" "http://127.0.0.1:${MINIO_PORT}/minio/health/live" 180

start_role mixcoord "$((METRICS_BASE + 1))"
start_role datanode "$((METRICS_BASE + 2))"
start_role querynode "$((METRICS_BASE + 3))"
start_role streamingnode "$((METRICS_BASE + 4))"
start_role proxy "$((METRICS_BASE + 5))"

wait_for_tcp "Milvus Proxy" "$MILVUS_HOST" "$MILVUS_PORT" 120
ensure_venv

log "Running QueryView PyMilvus smoke test"
"$VENV_DIR/bin/python" "$SMOKE_TEST" \
    --host "$MILVUS_HOST" \
    --port "$MILVUS_PORT" \
    2>&1 | tee "$LOG_DIR/pymilvus.log"

if [[ -n "$M1A_TEST_PLAN_DIR" ]]; then
    log "Preparing OpenAI 50K M1A collection"
    "$M1A_PYTHON" "$M1A_TEST_PLAN_DIR/scripts/prepare_openai_50k.py" \
        --host "$MILVUS_HOST" --port "$MILVUS_PORT" --shards-num "$M1A_SHARDS_NUM" \
        2>&1 | tee "$LOG_DIR/m1a-prepare.log"

    if [[ -n "$M1A_MEMORY_MODE" ]]; then
        proxy_pid="${MILVUS_PIDS[${#MILVUS_PIDS[@]}-1]}"
        log "Measuring Proxy memory for $M1A_MEMORY_MODE requests"
        "$M1A_PYTHON" "$M1A_TEST_PLAN_DIR/scripts/run_m1a_proxy_memory.py" \
            --mode "$M1A_MEMORY_MODE" --proxy-pid "$proxy_pid" \
            --proxy-metrics-url "http://127.0.0.1:$((METRICS_BASE + 5))/metrics_default" \
            --host "$MILVUS_HOST" --port "$MILVUS_PORT" \
            --concurrency "$M1A_MEMORY_CONCURRENCY" --top-k "$M1A_MEMORY_TOP_K" \
            --artifact-root "$RUN_DIR/m1a-memory" \
            2>&1 | tee "$LOG_DIR/m1a-memory-$M1A_MEMORY_MODE.log"
    else
        log "Running OpenAI 50K M1A correctness cases"
        "$M1A_PYTHON" "$M1A_TEST_PLAN_DIR/scripts/run_m1a_openai_correctness.py" \
            --host "$MILVUS_HOST" --port "$MILVUS_PORT" \
            --artifact-root "$RUN_DIR/m1a-artifacts" \
            2>&1 | tee "$LOG_DIR/m1a-correctness.log"
    fi
fi
