# Host Environment Notes

## Separation of Roles

- `AICAS/`: development and historical experiments
- `AICAS2026/V3/`: official submission and reproduction bundle

Do not mix temporary development outputs directly into `V3`.

## What Belongs in V3

- official scripts
- official documentation
- self-contained payload
- official results
- manifests and checksums
- source-state snapshot

## Build Notes

By default, `prepare_v3_bundle.sh` uses:

- SDK env: `/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux`
- build dir: `build-kv260-npu-current`

If a valid board binary already exists, the recommended path is:

```bash
bash AICAS2026/V3/scripts/prepare_v3_bundle.sh --skip-build
```

## Default Remote Settings

- host: `192.168.0.10`
- user: `ubuntu`
- remote root: `/home/ubuntu/aicas`
- sudo password: `123456`

If the organizer uses a different setup, override these values on the command line instead of editing the scripts.
