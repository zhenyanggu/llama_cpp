# Host Environment Policy

## Purpose

`AICAS/` stays the development workspace. `AICAS2026/V1/` is the submission workspace.

Do not mix the two.

## Host Layout

Development area:

- `AICAS/`
- `build-kv260-npu/`
- `build-kv260/`
- `npuruntime/kv260/`

Submission area:

- `AICAS2026/V1/docs/`
- `AICAS2026/V1/scripts/`
- `AICAS2026/V1/payload/`
- `AICAS2026/V1/results/official/`
- `AICAS2026/V1/manifests/`
- `AICAS2026/V1/source_state/`

Archive area:

- `AICAS2026/V1_412/`

## What Goes Into V1

Only the following classes of files belong in `AICAS2026/V1/`:

- official automation scripts
- official reproduction docs
- generated submission payload
- official run outputs
- generated manifests and checksums
- source snapshot metadata for the exact workspace used to build the binary

## What Must Stay Out Of V1

- ad hoc debug logs
- temporary JSON files named by ports
- variant experiments that are not part of the official `Q8_0 + mixed_v1` submission target
- full raw profiling dumps from exploratory work
- unrelated `AICAS/` scratch outputs

## Naming Rules

- Use `results/official/<UTC-run-id>/` for every official run.
- Use one stable result filename per artifact type:
  `board_readiness.txt`, `server.log`, `throughput_metrics.json`, `acc_100sample.json`, `run_meta.json`, `summary_metrics.json`, `RESULTS.md`
- Do not create new files named like `manual-throughput-8090.json` or `server-npu-summary.log` under `V1/`.

## Version Freeze

The official host snapshot is recorded in:

- `source_state/git_head.txt`
- `source_state/git_status.txt`
- `source_state/working_tree.patch`
- `source_state/untracked_files.txt`

Regenerate these files whenever the official run is rebuilt from a changed workspace.

## Normal Workflow

1. Make code changes under the main repo as usual.
2. Run `scripts/prepare_v1_bundle.sh`.
3. Run `scripts/run_v1_submission.sh`.
4. Inspect `results/official/<run_id>/`.
5. Keep `V1_412` unchanged unless you are explicitly archiving more exploratory material there.
