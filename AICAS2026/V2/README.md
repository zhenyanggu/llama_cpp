# AICAS2026 V2 (Double DMA Overlay)

V2 keeps the V1 remote layout constraint (`/home/ubuntu/aicas`) but stores local results under:

- `AICAS2026/V2/results/official/<run_id>/`

## Quick Flow

1. Initialize overlay app and NPU driver on KV260:

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
export BOARD_SUDO_PASSWORD='123456'
bash AICAS2026/V2/scripts/board_init_overlay.sh
```

2. Run throughput + 5-sample accuracy:

```bash
bash AICAS2026/V2/scripts/run_v2_eval.sh \
  --build-dir build-kv260-npu-current \
  --samples 5 \
  --skip-build \
  --skip-readiness-probe
```

3. If 5-sample result is normal, run 100-sample accuracy:

```bash
bash AICAS2026/V2/scripts/run_v2_eval.sh \
  --build-dir build-kv260-npu-current \
  --samples 100 \
  --skip-build \
  --skip-readiness-probe
```

## Notes

- Fixed text model: `AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf` (official `Q8_0`).
- Fixed mmproj: `AICAS/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf`.
- The current board app is `double_dma_overlayapp` (not `mynpu`).
- Because `xmutil listapps` behavior can vary by image/permission, V2 defaults to `--skip-readiness-probe` and relies on explicit board init.
- Treat `build-kv260-npu-current` as the canonical KV260 NPU build directory for V2 runs unless a run explicitly overrides it.
