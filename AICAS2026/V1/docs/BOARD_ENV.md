# Board Environment Policy

## Remote Root

All official V1 board-side files live under:

```text
/home/ubuntu/aicas
```

Nothing official should be written directly to `/home/ubuntu` except the user's normal shell files.

## Board Layout

Under `/home/ubuntu/aicas`:

- `board_support/mynpu/`
  Staged copies of `KV260_410.bin`, `KV260_410.dtbo`, and `shell.json`
- `board_support/driver/`
  Staged copy of `npu_kv260.ko`
- `shared/gguf/`
  Official model and mmproj files
- `shared/eval/`
  `throughput_eval.py`, `acc_eval.py`, `llama_server_client.py`, and `sampled_100.json`
- `shared/data/`
  Only the 100-sample minimal image set
- `shared/lib/`
  Runtime libraries needed by the KV260 binary
- `runs/submission-q8-w8a8/<run_id>/`
  One directory per official run

## System-Level Installation

The board initialization step copies the current `mynpu` app to:

```text
/lib/firmware/xilinx/mynpu
```

That is the location used by:

```bash
sudo xmutil loadapp mynpu
```

## Official Initialization

Run the official initialization only through:

```bash
/home/ubuntu/aicas/scripts/board_init_npu.sh
```

This script:

- installs `mynpu`
- unloads and reloads the app
- inserts `npu_kv260.ko`
- fixes `/dev/npu_kv260` ownership and permissions
- writes a readiness report

## Board Cleanliness Rules

- Do not place result JSON files directly under `/home/ubuntu`.
- Do not keep multiple renamed copies of the same official run in the same directory.
- Do not keep old binaries in `/home/ubuntu` with names like `llama-server-test`, `llama-server-new`, or `llama-server-final`.
- If a rerun is needed, create a new run id under `runs/submission-q8-w8a8/`.

## Python Policy

The current V1 evaluation path uses the board system `python3` only.

No extra board-side Python packages are required by:

- `throughput_eval.py`
- `acc_eval.py`
- `llama_server_client.py`

If future versions add extra dependencies, place the venv under:

```text
/home/ubuntu/aicas/venv/
```

and document the exact install command in this file before using it officially.
