# KV260 Throughput Eval Latency Breakdown

## Scope

This note captures the latency decomposition for the SmolVLM2 KV260 `throughput_eval.py` case using the source-built `llama.cpp` stack in this repository.

Model and assets:

- Text model: `SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- Multimodal projector: `mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- Image: `IIIT5K/test/2543_2.png`
- Threads: `-t 4`
- Alias: `smolvlm2-gguf`

## Raw Artifacts

- End-to-end throughput output: `throughput_check_prefill.log`
- Server log with multimodal processing timestamps: `server_check_prefill.log`
- Text ggml operator profile: `full_profile_v2.json`
- Text ggml per-node samples: `full_profile_v2_nodes.json`
- Multimodal projector ggml operator profile: `mmproj_profile.json`
- Runtime graph dumps:
  - `../ggml-dot-dumps/mmproj.dot`
  - `../ggml-dot-dumps/text_prefill.dot`
  - `../ggml-dot-dumps/text_decode.dot`
  - `../ggml-dot-dumps/mmproj.svg`
  - `../ggml-dot-dumps/text_prefill.sfdp.svg`
  - `../ggml-dot-dumps/text_decode.sfdp.svg`

## End-to-End Throughput Eval

From `throughput_check_prefill.log`:

- Prompt tokens: `188`
- Completion tokens: `202`
- Prefill time: `45736.15 ms`
- Decode time: `24144.96 ms`
- Prefill speed: `4.11 t/s`
- Decode speed: `8.37 t/s`

## Prefill Decomposition

From `server_check_prefill.log`:

- `image processed in 36498 ms`
- `prompt eval time = 45736.15 ms / 188 tokens`

From `mmproj_profile.json`:

- Pure `mmproj` graph time: `32018.068 ms`
- `MUL_MAT` inside `mmproj`: `30292.813 ms`

Derived split:

| Component | Time (ms) | Share of prefill |
| --- | ---: | ---: |
| Total prefill (`prompt_ms`) | 45736.150 | 100.00% |
| Multimodal section (`image processed`) | 36498.000 | 79.80% |
| `mmproj` encode only | 32018.068 | 70.01% |
| Image embeddings decoded into llama | 4479.932 | 9.80% |
| Remaining text prompt prefill after image | 9238.150 | 20.20% |
| `mmproj` `MUL_MAT` only | 30292.813 | 66.23% |

Interpretation:

- `throughput_eval` prefill **does include** `mmproj`.
- The dominant cost in prefill is the `mmproj` encode path.
- Inside `mmproj`, `MUL_MAT` dominates the graph.

## Text Model ggml Operator Profile

These numbers come from `full_profile_v2.json`, produced by node-by-node ggml replay through the `llama-mtmd-profiler` tool. Use them primarily as operator share breakdowns, not as replacements for server `prompt_ms` and `predicted_ms`.

### Text Prefill Graph

- Profiled text prefill total: `13578.014 ms`
- Top operators:

| Operator | Time (ms) | Share |
| --- | ---: | ---: |
| `MUL_MAT` | 12247.590 | 90.20% |
| `FLASH_ATTN_EXT` | 1048.340 | 7.72% |
| `SWIGLU` | 86.002 | 0.63% |
| `SET_ROWS` | 79.992 | 0.59% |
| `RMS_NORM` | 38.133 | 0.28% |
| `ROPE` | 35.632 | 0.26% |

Representative `MUL_MAT` signatures:

| Signature | Time (ms) | Share |
| --- | ---: | ---: |
| `q8_0 [960,2560] x f32 [960,115] -> f32 [2560,115]` | 3744.527 | 27.58% |
| `q8_0 [960,2560] x f32 [960,64] -> f32 [2560,64]` | 2085.735 | 15.36% |
| `q8_0 [2560,960] x f32 [2560,115] -> f32 [960,115]` | 1902.479 | 14.01% |
| `q8_0 [960,960] x f32 [960,115] -> f32 [960,115]` | 1453.570 | 10.71% |
| `q8_0 [2560,960] x f32 [2560,64] -> f32 [960,64]` | 1061.513 | 7.82% |

### Text Decode Graph

- Profiled text decode total: `26999.090 ms`
- Decode step count: `212`
- Average profiled decode step time: about `127.35 ms`
- Top operators:

| Operator | Time (ms) | Share |
| --- | ---: | ---: |
| `MUL_MAT` | 19161.313 | 70.97% |
| `FLASH_ATTN_EXT` | 5659.098 | 20.96% |
| `SWIGLU` | 541.700 | 2.01% |
| `RMS_NORM` | 349.233 | 1.29% |
| `ADD` | 317.995 | 1.18% |
| `ROPE` | 266.999 | 0.99% |

Representative `MUL_MAT` signatures:

| Signature | Time (ms) | Share |
| --- | ---: | ---: |
| `q8_0 [960,2560] x f32 [960,1] -> f32 [2560,1]` | 8477.099 | 31.40% |
| `q8_0 [2560,960] x f32 [2560,1] -> f32 [960,1]` | 4027.018 | 14.92% |
| `q8_0 [960,960] x f32 [960,1] -> f32 [960,1]` | 3122.822 | 11.57% |
| `q8_0 [960,49280] x f32 [960,1] -> f32 [49280,1]` | 2349.232 | 8.70% |
| `q8_0 [960,320] x f32 [960,1] -> f32 [320,1]` | 1185.142 | 4.39% |

## mmproj ggml Operator Profile

From `mmproj_profile.json`:

- Profiled `mmproj` total: `32018.068 ms`
- `MUL_MAT` total: `30292.813 ms`
- `MUL_MAT` share: `94.61%`

Top `mmproj` operators:

| Operator | Time (ms) | Share |
| --- | ---: | ---: |
| `MUL_MAT` | 30292.813 | 94.61% |
| `SOFT_MAX` | 921.492 | 2.88% |
| `GELU` | 323.242 | 1.01% |
| `CONT` | 208.128 | 0.65% |
| `ADD` | 175.226 | 0.55% |
| `NORM` | 67.131 | 0.21% |

Representative `mmproj` `MUL_MAT` signatures:

| Signature | Time (ms) | Share |
| --- | ---: | ---: |
| `q8_0 [3072,768] x f32 [3072,1024] -> f32 [768,1024]` | 6234.777 | 19.47% |
| `q8_0 [768,768] x f32 [768,1024] -> f32 [768,1024]` | 6229.460 | 19.46% |
| `q8_0 [768,3072] x f32 [768,1024] -> f32 [3072,1024]` | 6214.346 | 19.41% |
| `f32 [1024,64,12] x f32 [1024,1024,12] -> f32 [64,1024,12]` | 6085.900 | 19.01% |
| `f32 [64,1024,12] x f32 [64,1024,12] -> f32 [1024,1024,12]` | 5218.756 | 16.30% |
| `q8_0 [12288,960] x f32 [12288,64] -> f32 [960,64]` | 162.228 | 0.51% |

## Graph Dumps

Runtime graphs were dumped with `ggml_graph_dump_dot()` for three scopes:

- `mmproj.dot`: image encoder / projector graph
- `text_prefill.dot`: text-model batched prefill graph
- `text_decode.dot`: text-model single-token decode graph

The raw `.dot` and rendered `.svg` are kept in `../ggml-dot-dumps/`.

## Notes and Caveats

- `prompt_ms` from `llama-server` includes the multimodal path.
- `image processed` from the server log includes both `mmproj` encode and the subsequent image-embedding decode into the text model.
- The ggml operator profiles were collected by forcing node-by-node synchronized execution. Their totals are useful for decomposition, but should not replace the end-to-end throughput timings from `llama-server`.
- All text-side `MUL_MAT` signatures observed in this run were quantized-weight paths with `q8_0` weights and `f32` activations.
