#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

log() { printf '\n---- %s ----\n' "$*"; }

run_step() {
    local desc=$1; shift
    log "$desc"
    "$@"
}

# name:config:job-id  (":" separated, since job-ids/configs vary per case)
TS_JOBS=(
    "Eccentricity:Eccentricity/config_injection_example.ini:19526"
    "HoM:HoM/config_injection.ini:322"
    "Superimposed:Superimposed/config_injection.ini:98"
    "ExtremeSpin:ExtremeSpin/config_injection.ini:44"
)

TTMAP_JOBS=(
    "Eccentricity:Eccentricity/config_TT_map.ini:19526"
    "Precessing:Precessing/config_TT_map.ini:1"
    "HoM:HoM/config_TT_map.ini:322"
    "Superimposed:Superimposed/config_TT_map.ini:98"
    "ExtremeSpin:ExtremeSpin/config_TT_map.ini:44"
)

log "TS"

for entry in "${TS_JOBS[@]}"; do
    IFS=':' read -r name config job_id <<< "$entry"
    run_step "Generating the $name Example TS" \
        python "$name/TS_production.py" --config "$config" --job-id "$job_id"
done

# Precessing has a different script/signature, so handle it on its own
run_step "Generating the Precessing Example TS" \
    python Precessing/TS_production_batch.py --analysis injection --job-id 1 --type-inj mixed

log "TT MAP"

for entry in "${TTMAP_JOBS[@]}"; do
    IFS=':' read -r name config job_id <<< "$entry"
    run_step "Generating the $name TTmap" \
        python TT_map_opt_example.py --config "$config" --job-id "$job_id"
done
