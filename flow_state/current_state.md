# KV260 NPU performance workflow

## Goal

Improve 501 prompt prefill performance without changing correctness or precision.

## Bounds

- Scope: llama.cpp and NPU runtime software only.
- Correctness gate: 501 smoke profile run.
- Stop rule: stop after two consecutive valid candidate iterations improve less than 2% over the current best.
- Do not submit generated artifacts such as `.ko`, result directories, dumps, or transient logs.

## Baseline

- Main repo commit: faad577c9 NPU: overlap log8PV attention prep
- VersaEdge runtime/overlay commit: fdbba42 Versa_P: disable profile counters in prefill overlay
- Result directory: AICAS2026/aicas_semi/results/kv260/20260603T-act-cma-cpu-textpipe-501-final/results
- Prompt tokens: 501
- Prefill: 19903.351 ms, 25.172 tok/s
- mmproj encode: 6914.903 ms
- text prefill: 12974.854 ms
- Text attention: 480 rows, pipeline rows 480
- Text attention totals: kv_quant 1043.100 ms, q_quant 1227.086 ms, cpu_quant_overlap 1209.336 ms, npu_wait_after_quant 1070.961 ms, npu_call_wall 1192.660 ms, post_hidden 207.674 ms

## Current State

- State: STOPPED
- Candidate: A1 retry after driver ABI fix: bounded parallel CPU prepare workers for grouped text log8PV attention.
- Rationale: current single pending prepare group leaves about 1071 ms of NPU wait-after-quant in the 501 text attention profile. K/V cannot be merged across KV heads because the runtime group API shares one K/V buffer across the group, so the low-risk path is to prepare several future groups concurrently while preserving ordered NPU execution.
- Expected effect: reduce text attention `npu_wait_after_quant_us` and total text prefill wall time without changing quantization formulas, calibration, V centering, or log8 P format.

## Transition Log

- INIT_RECORD: baseline and stop criteria recorded.
- PROFILE_TRIAGE: selected text attention CPU prepare wait as first candidate.
- IMPLEMENT_CANDIDATE: added `AICAS_TEXT_LOG8PV_CPU_PREP_WORKERS`, default 2, with ordered bounded prepare queue and profile field `pipeline_workers`.
- LOCAL_VERIFY: `git diff --check`, `cmake --build build-kv260 --config Release -j8`, and `cmake --build build-kv260-semi --config Release -j8` passed.
- BOARD_501_SMOKE: first run failed before inference because `/dev/npu_kv260` did not appear after `xmutil loadapp`; platform device `a0000000.npu_generic` existed and compatible matched, but auto probe did not bind. Manual `driver_override=npu_kv260` plus bind recovered the device. Added the same fallback bind to the KV260 eval script.
- BOARD_501_SMOKE: A1 workers=2 run and workers=1 control both left only `throughput_eval.py` waiting after `llama-server` exited during the request, with no kernel segfault/OOM/NPU fault. This was later invalidated by the tracked driver/runtime ABI mismatch; A1 must be retested after the driver fix.
- LOCAL_VERIFY: rebuilt `build-kv260-semi` after reverting A1 attention changes.
- ENV_FIX: baseline-code with the tracked 567640-byte driver still left `throughput_eval.py` waiting after `llama-server` exited. Reusing the 583168-byte driver from the last successful run completed 501 smoke, proving a local tracked driver/runtime ABI mismatch. Replaced `npuruntime/kv260/driver/npu_kv260.ko` with the successful driver hash `448c327201fe07c285ded44497f3ef3a50c8a86527ab8edf621b678b4d21e858`.
- BOARD_501_SMOKE: default path with updated driver completed 501 smoke: 20571.799 ms, 24.354 tok/s. Treat as environment-repaired baseline; still slower than best 19903.351 ms, likely board-state variance.
- PROFILE_TRIAGE: selected A2 to reduce text log8PV CPU quant/dequant overhead without changing formulas or scheduling.
- IMPLEMENT_CANDIDATE: added F32/stride gates and replaced grouped fast-path per-element generic tensor get/set calls with direct row pointer loads/stores.
- LOCAL_VERIFY: `git diff --check` and `cmake --build build-kv260-semi --config Release -j8` passed.
- BOARD_501_SMOKE: A2 completed 501 smoke but regressed to 26753.422 ms. Profile showed text grouped pipeline disabled (`pipeline rows=0`) because the new F32 gate was not satisfied. Reverted A2 attention code.
- LOCAL_VERIFY: rebuilt `build-kv260-semi` after reverting A2 attention changes.
- BOARD_501_SMOKE: final default-path verification completed 501 smoke with updated tracked driver: 20608.181 ms, 24.311 tok/s. mmproj encode 6879.697 ms, text prefill 13718.834 ms. Text attention pipeline rows 480, q_quant 1616.074 ms, kv_quant 1274.861 ms, cpu_quant_overlap 1812.396 ms, npu_wait_after_quant 1090.854 ms.
- IMPLEMENT_CANDIDATE: reapplied A1 after updating the tracked driver. Added `AICAS_TEXT_LOG8PV_CPU_PREP_WORKERS`, default 2, with ordered bounded prepare queue and profile field `pipeline_workers`.
- LOCAL_VERIFY: `git diff --check` and `cmake --build build-kv260-semi --config Release -j8` passed for A1 retry.
- BOARD_501_SMOKE: A1 worker sweep completed. workers=2: 20390.308 ms; workers=3: 20236.574 ms; workers=4: 19997.172 ms. workers=4 improves 2.97% over the environment-repaired baseline 20608.181 ms, so set the default worker count to 4.
- LOCAL_VERIFY: `git diff --check` and `cmake --build build-kv260-semi --config Release -j8` passed after setting default workers to 4.
- BOARD_501_SMOKE: final default worker=4 run completed 501 smoke: 19994.848 ms, 25.056 tok/s. mmproj encode 6880.239 ms, text prefill 13104.937 ms. Text attention pipeline rows 480 with workers [4], q_quant 2175.706 ms, kv_quant 1916.260 ms, cpu_quant_overlap 3248.490 ms, npu_wait_after_quant 893.046 ms, npu_call_wall 1345.574 ms.
- STOP_OR_NEXT: stopped this iteration because the validated worker=4 change clears the 2% gate versus the repaired baseline, while further immediate candidates are either invalidated (direct F32 gate disabled grouped path) or require deeper dtype/layout investigation. Next state should investigate text attention K mvin variance and possible dtype-specialized quant loops before another code candidate.
