#!/usr/bin/env bash
set -u -o pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_regression.sh [options]

Run the KV260 NPU overlay regression suite entirely on the board.
The overlay app must already be installed for xmutil.

Options:
  --app-name <name>           xmutil app name (default: double_dma_overlayapp)
  --profile <fast|full|versa-p>
                               Test profile (default: full)
  --results-dir <path>        Results directory (default: <root>/runs/<timestamp>)
  --sudo-password-env <var>   Env var containing sudo password (default: KV260_SUDO_PASSWORD)
  --timeout <seconds>         Per-test timeout if timeout(1) exists (default: 300)
  --continue-on-fail          Run remaining tests after a failure
  -h, --help                  Show this help

Examples:
  ./scripts/run_regression.sh
  ./scripts/run_regression.sh --app-name my_overlay_app
  KV260_SUDO_PASSWORD=123456 ./scripts/run_regression.sh --profile fast
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$ROOT_DIR/bin"
DRIVER_KO="$ROOT_DIR/driver/npu_kv260.ko"
RUNS_DIR="$ROOT_DIR/runs"

APP_NAME="double_dma_overlayapp"
PROFILE="full"
RESULTS_DIR=""
SUDO_PASSWORD_ENV="KV260_SUDO_PASSWORD"
CASE_TIMEOUT="300"
CONTINUE_ON_FAIL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --app-name) APP_NAME="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --sudo-password-env) SUDO_PASSWORD_ENV="$2"; shift 2 ;;
    --timeout) CASE_TIMEOUT="$2"; shift 2 ;;
    --continue-on-fail) CONTINUE_ON_FAIL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  fast|full|versa-p) ;;
  *) echo "Unsupported profile: $PROFILE" >&2; exit 2 ;;
esac

if [ -z "$RESULTS_DIR" ]; then
  ts="$(date +%Y%m%dT%H%M%S%z)"
  RESULTS_DIR="$RUNS_DIR/$ts"
fi
mkdir -p "$RESULTS_DIR/cases"
SUMMARY_TSV="$RESULTS_DIR/summary.tsv"
META_JSON="$RESULTS_DIR/run_meta.json"
: > "$SUMMARY_TSV"
printf 'case\tstatus\texit_code\tduration_sec\tlog\n' >> "$SUMMARY_TSV"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

quote_json() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

app_pl_dtsi() {
  printf '/lib/firmware/xilinx/%s/pl.dtsi\n' "$APP_NAME"
}

app_compatibles() {
  local pl
  pl="$(app_pl_dtsi)"
  [ -f "$pl" ] || return 0
  sed -n 's/.*compatible[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$pl" | sort -u
}

driver_aliases() {
  if command -v modinfo >/dev/null 2>&1; then
    modinfo "$DRIVER_KO" 2>/dev/null | sed -n 's/^alias:[[:space:]]*//p'
  fi
}

check_driver_matches_app() {
  local pl compat saw_hw_compat=0 matched=0
  pl="$(app_pl_dtsi)"
  if [ ! -f "$pl" ]; then
    log "skip driver/app compatible check: missing $pl"
    return 0
  fi
  if ! command -v modinfo >/dev/null 2>&1; then
    log "skip driver/app compatible check: modinfo not found"
    return 0
  fi

  while IFS= read -r compat; do
    case "$compat" in
      ""|fixed-factor-clock|xlnx,afi-fpga|xlnx,fclk)
        continue
        ;;
    esac
    saw_hw_compat=1
    if driver_aliases | grep -Fq "$compat"; then
      matched=1
      break
    fi
  done < <(app_compatibles)

  if [ "$saw_hw_compat" -eq 0 ]; then
    log "skip driver/app compatible check: no hardware compatible found in $pl"
    return 0
  fi
  if [ "$matched" -eq 1 ]; then
    return 0
  fi

  {
    echo "Driver module does not match overlay app compatible strings."
    echo "app=$APP_NAME"
    echo "pl_dtsi=$pl"
    echo "app compatibles:"
    app_compatibles | sed 's/^/  /'
    echo "driver=$DRIVER_KO"
    echo "driver aliases:"
    driver_aliases | sed 's/^/  /'
  } >&2
  return 1
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return $?
  fi

  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return $?
  fi

  local password="${!SUDO_PASSWORD_ENV:-}"
  if [ -z "$password" ]; then
    echo "Missing sudo password in env var: $SUDO_PASSWORD_ENV" >&2
    echo "Either configure passwordless sudo or run: export $SUDO_PASSWORD_ENV='<password>'" >&2
    return 125
  fi

  printf '%s\n' "$password" | sudo -S -p '' "$@"
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "Missing $label: $path" >&2
    exit 2
  fi
}

require_executable() {
  local path="$1"
  local label="$2"
  if [ ! -x "$path" ]; then
    echo "Missing or non-executable $label: $path" >&2
    exit 2
  fi
}

collect_readiness() {
  local out="$1"
  {
    echo "== date =="
    date -Is
    echo
    echo "== app =="
    echo "$APP_NAME"
    echo
    echo "== app pl.dtsi =="
    pl="$(app_pl_dtsi)"
    if [ -f "$pl" ]; then
      echo "$pl"
      echo "compatibles:"
      app_compatibles | sed 's/^/  /'
    else
      echo "missing: $pl"
    fi
    echo
    echo "== driver module =="
    echo "$DRIVER_KO"
    if command -v modinfo >/dev/null 2>&1; then
      echo "aliases:"
      driver_aliases | sed 's/^/  /'
    else
      echo "modinfo not found"
    fi
    echo
    echo "== xmutil =="
    if [ -x /usr/bin/xmutil ]; then
      run_sudo xmutil listapps || true
    elif command -v xmutil >/dev/null 2>&1; then
      run_sudo xmutil listapps || true
    else
      echo "xmutil not found"
    fi
    echo
    echo "== dev =="
    ls -l /dev/npu_kv260 2>/dev/null || true
    echo
    echo "== modules =="
    grep '^npu_kv260 ' /proc/modules || true
    echo
    echo "== platform =="
    ls /sys/bus/platform/devices 2>/dev/null | grep -Ei 'npu|a0000000' || true
  } > "$out" 2>&1
}

assert_ready() {
  command -v xmutil >/dev/null 2>&1 || [ -x /usr/bin/xmutil ] || {
    echo "xmutil is not available" >&2
    return 1
  }
  xmutil_app_available || {
    echo "xmutil app is not visible: $APP_NAME" >&2
    return 1
  }
  [ -e /dev/npu_kv260 ] || {
    echo "/dev/npu_kv260 is missing" >&2
    return 1
  }
  grep -q '^npu_kv260 ' /proc/modules || {
    echo "npu_kv260 module is not loaded" >&2
    return 1
  }
  ls /sys/bus/platform/devices 2>/dev/null | grep -Eiq 'npu|a0000000' || {
    echo "NPU platform device is missing" >&2
    return 1
  }
}

xmutil_app_available() {
  run_sudo xmutil listapps 2>/dev/null | grep -Fq "$APP_NAME"
}

overlay_platform_present() {
  ls /sys/bus/platform/devices 2>/dev/null | grep -Eiq 'npu|a0000000|Versa_P'
}

load_overlay_app() {
  local tmp rc
  tmp="$(mktemp)"

  run_sudo xmutil unloadapp || true

  run_sudo xmutil loadapp "$APP_NAME" > "$tmp" 2>&1
  rc=$?
  cat "$tmp"
  rm -f "$tmp"

  if [ "$rc" -eq 0 ]; then
    return 0
  fi

  if xmutil_app_available && overlay_platform_present; then
    log "xmutil loadapp returned rc=$rc but $APP_NAME platform device is present; continuing"
    return 0
  fi

  return "$rc"
}

reload_overlay_and_driver() {
  log "reload overlay app=$APP_NAME"

  if grep -q '^npu_kv260 ' /proc/modules; then
    run_sudo rmmod npu_kv260 || true
  fi

  load_overlay_app || return $?

  if grep -q '^npu_kv260 ' /proc/modules; then
    run_sudo rmmod npu_kv260 || true
  fi
  check_driver_matches_app || return $?
  run_sudo insmod "$DRIVER_KO" || return $?

  if [ -e /dev/npu_kv260 ]; then
    run_sudo chgrp "$(id -gn)" /dev/npu_kv260 || true
    run_sudo chmod 660 /dev/npu_kv260 || true
  fi

  assert_ready
}

run_with_optional_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${CASE_TIMEOUT}s" "$@"
  else
    "$@"
  fi
}

collect_dmesg_tail() {
  local out="$1"
  if command -v dmesg >/dev/null 2>&1; then
    run_sudo dmesg | tail -n 160 > "$out" 2>&1 || true
  fi
}

run_case() {
  local name="$1"
  shift
  local log_file="$RESULTS_DIR/cases/${name}.log"
  local ready_file="$RESULTS_DIR/cases/${name}.readiness.txt"
  local dmesg_file="$RESULTS_DIR/cases/${name}.dmesg_tail.txt"
  local start end rc status duration

  start="$(date +%s)"
  log "case start: $name"
  (
    step_rc=0
    echo "== case =="
    echo "$name"
    echo "== command =="
    printf '%q ' "$@"
    echo
    echo "== reload =="
    reload_overlay_and_driver || step_rc=$?
    echo
    echo "== readiness =="
    collect_readiness "$ready_file"
    cat "$ready_file"
    echo
    if [ "$step_rc" -ne 0 ]; then
      echo "reload/readiness failed rc=$step_rc"
      exit "$step_rc"
    fi
    echo "== output =="
    run_with_optional_timeout "$@"
  ) > "$log_file" 2>&1
  rc=$?
  end="$(date +%s)"
  duration=$((end - start))

  if [ "$rc" -eq 0 ]; then
    status="PASS"
  else
    status="FAIL"
    collect_dmesg_tail "$dmesg_file"
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$rc" "$duration" "$log_file" >> "$SUMMARY_TSV"
  log "case end: $name status=$status rc=$rc duration=${duration}s"

  if [ "$rc" -ne 0 ] && [ "$FAIL" -eq 0 ]; then
    FAIL="$rc"
  fi

  if [ "$rc" -ne 0 ] && [ "$CONTINUE_ON_FAIL" -ne 1 ]; then
    return "$rc"
  fi
  return 0
}

require_file "$DRIVER_KO" "driver module"
require_executable "$BIN_DIR/kv260_npu_smoke_test" "smoke test"
require_executable "$BIN_DIR/kv260_runtime_init_test" "runtime init test"
require_executable "$BIN_DIR/kv260_dma_loopback_test" "DMA loopback test"

if [ "$PROFILE" = "full" ]; then
  require_executable "$BIN_DIR/kv260_dma_acc_int32_fp32_test" "DMA ACC int32/fp32 test"
  require_executable "$BIN_DIR/kv260_mvin_problem_case_test" "MVIN problem case test"
  require_executable "$BIN_DIR/kv260_dma_double_mvin_async_test" "double MVIN async test"
  require_executable "$BIN_DIR/kv260_layer_gemm_replay_test" "GEMM replay test"
fi
if [ "$PROFILE" = "versa-p" ]; then
  require_executable "$BIN_DIR/kv260_gemm_plan_test" "Versa_P GEMM plan test"
fi

cat > "$META_JSON" <<META
{
  "app_name": $(quote_json "$APP_NAME"),
  "profile": $(quote_json "$PROFILE"),
  "root_dir": $(quote_json "$ROOT_DIR"),
  "results_dir": $(quote_json "$RESULTS_DIR"),
  "driver": $(quote_json "$DRIVER_KO"),
  "started_at": $(quote_json "$(date -Is)"),
  "uname": $(quote_json "$(uname -a)")
}
META

log "results: $RESULTS_DIR"
log "profile: $PROFILE app: $APP_NAME"

FAIL=0
run_case readiness /bin/true || FAIL=$?
run_case smoke_1m "$BIN_DIR/kv260_npu_smoke_test" 1M || FAIL=$?
run_case runtime_init env NPU_CMA_SIZE=256M "$BIN_DIR/kv260_runtime_init_test" || FAIL=$?
if [ "$PROFILE" != "versa-p" ]; then
  run_case dma_loopback env NPU_CMA_SIZE=16M "$BIN_DIR/kv260_dma_loopback_test" || FAIL=$?
fi

if [ "$PROFILE" = "versa-p" ]; then
  run_case gemm_plan_mmproj_tail env NPU_CMA_SIZE=64M NPU_GEMM_PLAN_SKIP_REG_IO=1 NPU_GEMM_PLAN_ONLY=mmproj_tail_b64_128x64x64 "$BIN_DIR/kv260_gemm_plan_test" || FAIL=$?
  run_case gemm_plan_mmproj_full env NPU_CMA_SIZE=64M NPU_GEMM_PLAN_SKIP_REG_IO=1 NPU_GEMM_PLAN_ONLY=mmproj_full_128x240x64 "$BIN_DIR/kv260_gemm_plan_test" || FAIL=$?
fi

if [ "$PROFILE" = "full" ]; then
  run_case smoke_256m "$BIN_DIR/kv260_npu_smoke_test" 256M || FAIL=$?
  run_case mvin_problem_case "$BIN_DIR/kv260_mvin_problem_case_test" --loops 100 || FAIL=$?
  run_case dma_2d_submatrix "$BIN_DIR/kv260_mvin_problem_case_test" --loops 20 --col 15 --row 15 --sram-stride 16 --dram-stride 64 || FAIL=$?
  run_case dma_acc_int32_fp32 "$BIN_DIR/kv260_dma_acc_int32_fp32_test" --loops 5 || FAIL=$?
  run_case dma_acc_fp32_perchannel "$BIN_DIR/kv260_dma_acc_int32_fp32_test" --loops 2 --col 31 --row 3 --sram-stride 32 --dram-stride 32 --scale 0.75 --zero-point 0 --per-channel || FAIL=$?
  run_case double_mvin_async "$BIN_DIR/kv260_dma_double_mvin_async_test" --loops 100 || FAIL=$?
  run_case gemm_basic "$BIN_DIR/kv260_layer_gemm_replay_test" --m 16 --n 16 --k 768 --loops 1 --no-double-mvin || FAIL=$?
  run_case gemm_double_dma "$BIN_DIR/kv260_layer_gemm_replay_test" --m 16 --n 16 --k 768 --loops 1 --double-mvin || FAIL=$?
  run_case gemm_fp32_tensor "$BIN_DIR/kv260_layer_gemm_replay_test" --m 16 --n 16 --k 768 --loops 1 --double-mvin --mvout-fp32 --fp32-scale 0.03125 --fp32-zp 0 || FAIL=$?
fi

log "summary: $SUMMARY_TSV"
cat "$SUMMARY_TSV"

summary_fail_count="$(awk -F '\t' 'NR > 1 && $2 != "PASS" { count++ } END { print count + 0 }' "$SUMMARY_TSV")"
if [ "$summary_fail_count" -eq 0 ]; then
  log "regression PASS"
  exit 0
else
  if [ "$FAIL" -eq 0 ]; then
    FAIL=1
  fi
  log "regression FAIL failed_cases=$summary_fail_count rc=$FAIL"
  exit "$FAIL"
fi
