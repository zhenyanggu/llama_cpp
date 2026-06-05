# ggml-npu architecture

This backend is split into small implementation fragments but is still built as
one translation unit through `ggml-npu.cpp`. That keeps the current internal
`static` helpers and caches private while making the code reviewable. Do not add
new large logic directly to `ggml-npu.cpp`; add it to the matching fragment.

## File layout

- `ggml-npu.cpp`: public includes, decode stream ABI mirror, fragment includes,
  and `GGML_BACKEND_DL_IMPL`.
- `ggml-npu-private-runtime.inc`: environment switches, decode runtime
  lifecycle, CMA/cache ownership, stream device handle, and the only wrapper
  around `npu_stream_gemv_run`.
- `ggml-npu-private-decode-helpers.inc`: W4/W8 layout, packing, profile, repro
  dump, and attention cache helpers.
- `ggml-npu-private-backend.inc`: ggml backend buffer/device/registration
  internals.
- `ggml-npu-api-backend.inc`: exported backend glue, memory helpers, shutdown,
  overlay state, and log8pv attention bridge.
- `ggml-npu-api-prefill.inc`: prefill/raw INT8 GEMM API.
- `ggml-npu-api-decode-linear.inc`: decode preload and W4/W8 linear GEMV entry
  points.
- `ggml-npu-api-decode-ffn-attn.inc`: decode W4 SwiGLU FFN and attention entry
  points.

`CMakeLists.txt` uses an explicit source list. The `.inc` fragments must not be
compiled as standalone sources.

## Decode stream contract

The llama.cpp decode integration is stream-only. Hardware decode execution must
go through:

```text
npu_decode_stream_gemv -> npu_stream_gemv_run
```

Do not call old decode MVIN/GEMV/MVOUT, matvec, ping-pong, or decode-flow
runtime APIs from llama.cpp. Old fallback call sites are routed to
`npu_decode_stream_unsupported_*`; reaching one is a bug in the stream path, not
permission to re-enable legacy runtime calls.

The public llama-facing symbols are intentionally kept stable, including W4/W8
preload, W4/W8 GEMV, W4 SwiGLU FFN, W4 attention, runtime shutdown, and overlay
active/inactive helpers. Their implementations must either run stream
descriptors or fail closed with an explicit log.

## Maintenance rules

- Add a stream descriptor builder or helper before adding another ad hoc
  descriptor fill block.
- Keep CMA allocation ownership local and explicit. If a function allocates CMA,
  it must free it on every return path.
- Keep profile and repro dump code opt-in. Do not enable prefill profile by
  default for decode debugging.
- W16 decode is not supported by the current stream contract. Keep it
  fail-closed unless runtime and stream tests prove otherwise.
- Before claiming RTL/runtime correctness, reproduce the failing operator in
  `sw/kv260/runtime/tests/stream_gemv_api_test.cpp`.

## Validation checklist

Use these checks after decode changes:

```bash
rg -n "npu_decode_(dma_mvin|dma_mvout|matvec|flow_make|mvin\\(|gemv\\(|mvout\\()|split_or_legacy" \
  ggml/src/ggml-npu/*.cpp ggml/src/ggml-npu/*.inc ggml/src/ggml-npu/*.h
rg -n "npu_stream_gemv_run" \
  ggml/src/ggml-npu/*.cpp ggml/src/ggml-npu/*.inc ggml/src/ggml-npu/*.h
cmake --build build-npu-refactor-check --target ggml-npu -j$(nproc)
```

The first command should not find active old decode runtime calls. The second
command should show the stream bridge and the ABI declaration.

Known current debug state: random dense W4 stream runtime testing passes
`attn_q_960x960` and `ffn_down_960x2560`, but fails several other W4 shapes.
This refactor must not hide that failure by falling back to legacy paths.
