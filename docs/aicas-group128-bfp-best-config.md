# AICAS Group128 BFP Best Config

This note records the current best local candidate from the group-128 AWQ,
mmproj SmoothQuant, and attention BFP search.

Last updated: 2026-05-18.

## Official FP16 Baseline

- Text model:
  `/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf`
- mmproj:
  `/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf`
- Stratified 100case score:
  `47/100`

## Best Quantized Candidate

- Text model:
  `AICAS/output/group128-hparam-search/text_awq/calib16-a0_12-g128/text_sq_a05_decode_awq_calib16_a0_12_g128_scale_f16.gguf`
- mmproj:
  `AICAS/output/group128-hparam-search/mmproj_sq/a0_7-minmax-per_channel/mmproj.gguf`
- Text prefill SmoothQuant alpha:
  `0.5`
- Text decode AWQ:
  `alpha=0.12`, `group_size=128`
- mmproj SmoothQuant:
  `alpha=0.70`, `per_channel`
- mmproj attention BFP16-M:
  full QK/PV, static per-layer/per-op int8 exponent table
- Text prefill attention BFP16-M:
  full QK/PV, static per-tensor exponent mode
- Text prefill static exponents:
  `Q=-11`, `K=-11`, `V=-12`, `P=-15`

## Runtime Environment

```bash
AICAS_TEXT_PREFILL_ATTN_BFP16M=1
AICAS_TEXT_PREFILL_BFP16M_EXP_MODE=static_tensor
AICAS_TEXT_PREFILL_BFP16M_Q_EXP=-11
AICAS_TEXT_PREFILL_BFP16M_K_EXP=-11
AICAS_TEXT_PREFILL_BFP16M_V_EXP=-12
AICAS_TEXT_PREFILL_BFP16M_P_EXP=-15
AICAS_MMPROJ_ATTN_PRECISION=bfp16m
AICAS_MMPROJ_ATTN_PRECISION_SCOPE=core
AICAS_MMPROJ_BFP16M_EXP_MODE=static_per_layer
```

## Stratified 100case Results

- Current best candidate, mmproj static BFP16-M plus text prefill static
  per-tensor BFP16-M:
  `52/100`
- Previous best candidate, mmproj static BFP16-M plus text prefill dynamic
  BFP16-M:
  `50/100`
- Candidate with BFP off:
  `47/100`
- Older candidate, AWQ `0.125` + mmproj SmoothQuant `0.50` + BFP:
  `40/100`

Current best result:
`AICAS/output/group128-hparam-search/retest-seed20260516/best_awq012_mm070_static_bfp16m_textprefill_static_tensor/SmolVLM2.json`

Evaluation set:
`AICAS/output/group128-hparam-search/stratified100-seed20260516-excl-prev-calib.json`

Current best by type:

- Regular Text Recognition: `10/10`
- Irregular Text Recognition: `4/10`
- Artistic Text Recognition: `7/10`
- Handwriting Recognition: `6/10`
- Digit String Recognition: `8/10`
- Non-Semantic Text Recognition: `4/10`
- Scene Text-centric VQA: `13/40`

## Best BFP8 Candidate

This is the current best BFP8 attention candidate. It is recorded separately
from the overall best BFP16-M candidate above.

- Text model:
  `AICAS/output/group128-hparam-search/text_awq/calib16-a0_12-g128/text_sq_a05_decode_awq_calib16_a0_12_g128_scale_f16.gguf`
- mmproj:
  `AICAS/output/group128-hparam-search/mmproj_sq/a0_7-minmax-per_channel/mmproj.gguf`
- mmproj attention:
  BFP8-M tile scale, `tile=32`
- Scale mapping:
  A side `[tile_m=32, K]` shares one scale; B side `[K, tile_n=32]`
  shares one scale.
- Text attention:
  normal evaluation keeps flash attention enabled; text prefill BFP8 custom
  QK/PV is not used on this path.

Runtime environment:

```bash
AICAS_TEXT_PREFILL_ATTN_BFP8M=1
AICAS_TEXT_PREFILL_BFP8M_SCALE_MODE=tile
AICAS_TEXT_PREFILL_BFP8M_TILE=32
AICAS_MMPROJ_ATTN_PRECISION=bfp8m
AICAS_MMPROJ_ATTN_PRECISION_SCOPE=core
AICAS_MMPROJ_BFP8M_SCALE_MODE=tile
AICAS_MMPROJ_BFP8M_TILE=32
AICAS_BFP8M_TILE=32
```

Stratified 100case result on the seed20260517 batch:

- BFP8 tile32 candidate:
  `47/100`
- Previous BFP8 K64 block candidate on the same batch:
  `46/100`
- Official FP16 baseline on the same batch:
  `44/100`

Best BFP8 result:
`AICAS/output/group128-hparam-search/bfp8m_attn/text-mmproj-bfp8m-tile32-seed20260517-100/SmolVLM2.json`

Best BFP8 evaluation set:
`AICAS/output/group128-hparam-search/stratified100-seed20260517-excl-rotated.json`

Best BFP8 by type:

- Regular Text Recognition: `7/10`
- Irregular Text Recognition: `5/10`
- Artistic Text Recognition: `8/10`
- Handwriting Recognition: `6/10`
- Digit String Recognition: `5/10`
- Non-Semantic Text Recognition: `4/10`
- Scene Text-centric VQA: `12/40`

Diagnostic caveat:

- Forcing `LLAMA_ARG_FLASH_ATTN=off` so text prefill QK/PV also uses tile32
  BFP8-M gave `0/30`; do not use tile32 BFP8 for text prefill custom attention
  as the current strategy.

## Kept Local Model Files

Only the official FP16 models and the two best candidate GGUF files above are
needed for the current comparison. The current best runtime knobs are recorded
in `docs/aicas-current-best-config.json`. The current best BFP8 runtime knobs
are recorded in `docs/aicas-current-best-bfp8-config.json`. Other GGUF files
generated under `AICAS/output/group128-hparam-search` were cleanup artifacts
from the alpha sweep and were removed.
