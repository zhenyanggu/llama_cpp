# AICAS Group128 BFP Best Config

This note records the current best local candidate from the group-128 AWQ and
mmproj SmoothQuant search.

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

## Stratified 100case Results

- FP16 official baseline:
  `47/100`
- Best candidate with BFP off:
  `47/100`
- Best candidate with BFP on:
  `46/100`
- Older candidate, AWQ `0.125` + mmproj SmoothQuant `0.50` + BFP:
  `40/100`

## Kept Local Model Files

Only the official FP16 models and the two best candidate GGUF files above are
needed for the current comparison. Other GGUF files generated under
`AICAS/output/group128-hparam-search` were cleanup artifacts from the alpha
sweep and were removed.
