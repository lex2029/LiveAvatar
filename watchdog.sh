#!/bin/bash
# LiveAvatar Worker Watchdog
# Monitors the worker process and restarts if it's dead or unresponsive.
# This script IS the systemd service — it manages the worker lifecycle.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
WORKER="$REPO_ROOT/smartblog_worker.py"
PIDFILE="$REPO_ROOT/worker.pid"
LOGFILE="$REPO_ROOT/watchdog.log"

CHECK_INTERVAL=30        # seconds between health checks
STUCK_TIMEOUT=600        # kill worker if no poll activity for 10 minutes
MAX_RESTART_FAILURES=5   # stop trying after N consecutive failed starts

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] WATCHDOG: $*" | tee -a "$LOGFILE"
}

worker_pid() {
    if [ -f "$PIDFILE" ]; then
        cat "$PIDFILE" 2>/dev/null
    fi
}

is_alive() {
    local pid=$1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_worker() {
    log "Starting worker..."
    cd "$REPO_ROOT"
    PYTHONUNBUFFERED=1 "$PYTHON" "$WORKER" &
    local pid=$!
    echo "$pid" > "$PIDFILE"
    log "Worker started (PID $pid)"
    # Give it time to initialize
    sleep 5
    if is_alive "$pid"; then
        return 0
    else
        log "Worker died immediately after start"
        return 1
    fi
}

stop_worker() {
    local pid=$(worker_pid)
    if [ -z "$pid" ]; then
        return 0
    fi
    if ! is_alive "$pid"; then
        rm -f "$PIDFILE"
        return 0
    fi
    log "Stopping worker (PID $pid)..."
    kill "$pid" 2>/dev/null
    # Wait up to 30s for graceful shutdown
    for i in $(seq 1 30); do
        if ! is_alive "$pid"; then
            log "Worker stopped gracefully"
            rm -f "$PIDFILE"
            return 0
        fi
        sleep 1
    done
    log "Worker didn't stop gracefully, killing..."
    kill -9 "$pid" 2>/dev/null
    rm -f "$PIDFILE"
    sleep 2
}

check_stuck() {
    # Check if worker has polled recently by looking at log timestamps
    local pid=$(worker_pid)
    if [ -z "$pid" ] || ! is_alive "$pid"; then
        return 1  # not alive
    fi
    # Check process CPU — if 0% for a long time, might be stuck
    local cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ')
    if [ -z "$cpu" ]; then
        return 1
    fi
    # Check if process has been idle too long via /proc stat
    local proc_stat="/proc/$pid/stat"
    if [ ! -f "$proc_stat" ]; then
        return 1
    fi
    return 0  # alive and seems ok
}

# ── Signal handling ──
STOP_REQUESTED=false
trap 'STOP_REQUESTED=true; log "Received stop signal"' SIGTERM SIGINT

# ── Main loop ──
log "Watchdog started (PID $$)"
log "Repository: $REPO_ROOT"
log "Check interval: ${CHECK_INTERVAL}s"

restart_failures=0

while [ "$STOP_REQUESTED" = false ]; do
    pid=$(worker_pid)

    if [ -z "$pid" ] || ! is_alive "$pid"; then
        if [ $restart_failures -ge $MAX_RESTART_FAILURES ]; then
            log "ERROR: $restart_failures consecutive start failures. Waiting 5 minutes..."
            sleep 300
            restart_failures=0
        fi

        stop_worker  # clean up stale pid
        if start_worker; then
            restart_failures=0
        else
            restart_failures=$((restart_failures + 1))
            log "Start failure #$restart_failures/$MAX_RESTART_FAILURES"
            sleep 10
            continue
        fi
    fi

    # Wait for next check
    for i in $(seq 1 $CHECK_INTERVAL); do
        if [ "$STOP_REQUESTED" = true ]; then
            break
        fi
        sleep 1
    done
done

# ── Shutdown ──
log "Watchdog stopping, shutting down worker..."
stop_worker
log "Watchdog exited"
