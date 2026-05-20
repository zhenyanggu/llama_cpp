# AICAS mmproj BFP8 Q8.24 KV260 profile, 2026-05-20

## Configuration

- Overlay app: `versa_prefill_profile_app`
- Best config: `docs/aicas-current-best-config.json`
- Text model: `/home/gugugu/work/model-quant/AICAS/output/group128-hparam-search/text_awq/calib16-a0_12-g128/text_sq_a05_decode_awq_calib16_a0_12_g128_scale_f16.gguf`
- mmproj: `/home/gugugu/work/model-quant/AICAS/output/group128-hparam-search/mmproj_sq/a0_7-minmax-per_channel/mmproj.gguf`
- mmproj precision: `AICAS_MMPROJ_ATTN_PRECISION=bfp8m`
- scale mode: `AICAS_MMPROJ_BFP8M_SCALE_MODE=tile_q8_24`
- NPU scope: mmproj only, with text W8A8 offload skipped via `MTMD_NPU_W8A8_SKIP_CONTAINS=weight`
- CMA2D path: disabled, `AICAS_MMPROJ_BFP8M_NPU_CMA2D=0`

The board loaded `/lib/firmware/xilinx/versa_prefill_profile_app`.

## Fix required to run

The first raw-packed mmproj qk tile was:

```text
mmproj_bfp8m_qk tile n=128 m=240 k=64 a_stride=64 b_stride=256
```

Before the fix, `ggml_backend_npu_i8_gemm_raw_packed()` submitted MVIN as a one-row byte stream. The Versa_P runtime interpreted that as `mvin_a.k = 8192`, causing:

```text
versa_p_start_mvin_a failed: illegal shape (-7)
```

The fixed path submits packed A and W as structured 2D tensors:

- A: `[cur_n, k]`, stride `act_stride_k`
- W: `[k, cur_m]`, stride `tile_m_stride`

## Profile run

Run id:

```text
20260520T-mmproj-bfp8-q824-versa-profile64-fixedmvin
```

Local results:

```text
AICAS2026/aicas_semi/results/kv260/20260520T-mmproj-bfp8-q824-versa-profile64-fixedmvin/results
```

Server startup confirmed:

```text
clip_ctx: AICAS mmproj BFP8-M k_block=64 tile=32 scale_mode=tile_q8_24 scale_shift=i32+i32
clip_ctx: AICAS mmproj BFP8-M NPU raw int8 GEMM enabled (required)
clip_ctx: CLIP using NPU backend
```

Throughput/profile output, 64 generated tokens:

```text
Prompt Tokens:     501
Completion Tokens: 64
Prefill Time:      92668.36 ms
Decode Time:       101023.88 ms
Total Time:        193692.25 ms
Prefill Speed:     5.41 t/s
Decode Speed:      0.63 t/s
```

The generated text was non-empty and structurally matched the prompt:

```text
Phase 1: Granular Visual Deconstruction
Phase 2: Semantic and Thematic Extractio
Phase 3: Phase 4: Phase 5: Phase 6: Phase 7: Phase 8: Phase 9: Phase 10: Phase 11: Phase 12
```

## Latency decomposition

From `mtmd_prefill_summary.json`:

| Phase | Time |
| --- | ---: |
| Total multimodal prefill | 92.668 s |
| mmproj encode | 49.886 s |
| merged text prefill | 42.768 s |
| mmproj graph compute | 49.869 s |
| graph build + alloc + inputs + readback | 0.015 s |

The mmproj graph assignment is still CPU-backend scheduled because the custom attention operator calls the raw NPU GEMM path internally. The largest scheduled categories by volume are:

| Category | Nodes | Bytes |
| --- | ---: | ---: |
| `MAP_CUSTOM3_CPU` | 97 | 981.7 MB |
| `ATTENTION_CPU` | 12 | 604.0 MB |
| `ELEMENTWISE_CPU` | 148 | 578.8 MB |
| `LAYOUT_CPU` | 122 | 383.0 MB |
| `ACTIVATION_CPU` | 12 | 151.0 MB |

The BFP8 custom NPU path also emitted an internal qk/pv breakdown under `bfp8m_npu`:

| Item | Value |
| --- | ---: |
| qk custom calls | 12 |
| pv custom calls | 12 |
| qk raw GEMM calls | 144 |
| pv raw GEMM calls | 144 |
| fallback calls | 0 |
| total custom qk/pv time | 27.887 s |
| qk custom time | 17.164 s |
| pv custom time | 10.723 s |
| A scale + quantize | 1.074 s |
| B scale + quantize | 8.849 s |
| NPU pack | 0.261 s |
| raw GEMM path | 8.666 s |
| postprocess | 8.989 s |
| raw int32 accumulator traffic | 641.7 MB |

From `text_cpu_profile.jsonl`, request-level decode is also a major bottleneck:

| Phase | Wall time |
| --- | ---: |
| merged text prefill | 42.762 s |
| 63 token decode records | 100.852 s |

Top text CPU operators:

| Phase | Operator | Time |
| --- | --- | ---: |
| merged prefill | `MAP_CUSTOM3` | 34.522 s |
| merged prefill | `FLASH_ATTN_EXT` | 7.514 s |
| decode | `MAP_CUSTOM3` | 94.891 s |
| decode | `FLASH_ATTN_EXT` | 2.458 s |

## Bottleneck assessment

For mmproj attention, the NPU currently accelerates only the raw qk/pv int8 GEMM pieces. The enclosing BFP8 attention remains dominated by CPU-side custom operator work and memory traffic: A/B scale search and quantization, packed-buffer preparation, per-call MVIN/GEMM/MVOUT synchronization inside `raw_gemm_us`, Q8.24 scale application, raw int32 readback, layout work, softmax/probability processing, and output postprocess. The custom path emits coarse qk/pv breakdowns in `mtmd_prefill_summary.json`, but it does not emit the generic `ggml_npu_profile.json`; exact per-tile NPU busy time versus host overhead still needs a dedicated raw-GEMM profiler.

For end-to-end throughput, mmproj is not the only bottleneck. With this text model/config, CPU text decode is slower than mmproj encode for the 64-token request and limits decode throughput to about `0.63 t/s`.

## Suggested fixes

Software-side:

- Add raw BFP8 mmproj NPU profiling around pack, host copy, MVIN A, MVIN W, GEMM, MVOUT, and CPU postprocess.
- Re-enable and validate the direct CMA2D/strided activation path after the RTL/runtime issue is fixed, to remove the current host repack path.
- Cache or pre-pack stable K/V or weight tiles where model layout permits.
- Reduce per-tile synchronization by double-buffering MVIN/GEMM/MVOUT.
- Move more of text AWQ decode off CPU if the measured target is full throughput, not just image prefill.

NPU/RTL features that would directly address the bottleneck:

- A fused attention primitive: QK, scale, mask, softmax, and PV with on-chip intermediate storage.
- Native BFP8/Q8.24 metadata support, including per-row-tile/per-column-tile scale application in MVOUT or GEMM epilogue.
- Strided/2D activation DMA that can feed GEMM without CPU repacking.
- Persistent/on-chip tile reuse for attention K/V or B tiles, avoiding repeated weight MVIN.
- Exposed per-command profile counters for MVIN A/W/meta, GEMM, MVOUT, stalls, and overlap, usable by this raw custom path.
