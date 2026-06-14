# Decode Runtime Snapshot

This directory is the llama.cpp-local snapshot of the KV260 decode stream
runtime used by the NPU backend build.

Source checkout:

- Path: `/mnt/c/vivado/versa_decode/sw/kv260/runtime`
- Repository HEAD: `97a4824b422c2185ccca643e0780b78483f97e6c`
- Branch at copy time: `decode`
- Copied on: 2026-06-06

Important: the copied runtime includes source-tree changes that were dirty in
the source checkout at copy time. In particular, this snapshot includes the
stream KV quant API additions:

- `NPU_STREAM_GEMV_F_KV_QUANT`
- `NPU_STREAM_GEMV_F_KV_IS_V`
- `npu_stream_kv_scale_result`
- `npu_stream_kv_scale_read()`

The source runtime diff hash for the stream header/runtime pair at copy time
was:

```text
0bb26cf9aa1c34784f1e5c374f0e8dcf788967873d7a11db55311f5e32df4dd2
```

Do not rely on `/mnt/c/vivado/versa_decode/sw/kv260/runtime` implicitly for
llama.cpp builds. Override `GGML_NPU_DECODE_RUNTIME_ROOT` only when deliberately
testing a newer external runtime.
