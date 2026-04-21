# Host Environment Notes

`VersaVLM/` is a standalone delivery directory.

The organizer only needs:

- `ssh`
- `scp`
- `python3`
- a complete `VersaVLM/` folder

## Host-Side Rules

- run all commands from the `VersaVLM/` root
- do not depend on the original development repository
- do not depend on external model directories
- do not depend on external evaluation script directories

## Self-Check Script

```bash
bash scripts/validate_bundle.sh
```

This validates:

- model
- mmproj
- overlay
- driver
- runtime libs
- 100-sample json
- throughput input image
- test binaries
- bilingual documentation
