# Repository Guidelines

## Project Structure & Module Organization
`src/`, `include/`, and `common/` hold the core C/C++ library and shared helpers. `ggml/` contains the tensor engine and backend implementations such as `ggml/src/ggml-cuda` and `ggml/src/ggml-vulkan`. User-facing binaries live under `tools/` (`server`, `main`, `quantize`, `llama-bench`, etc.), while `examples/` contains smaller reference programs. Tests are primarily in `tests/`, with some tool-specific scripts such as `tools/*/tests.sh`. Python conversion utilities live at the repo root and in `gguf-py/`. Documentation, helper scripts, and CI entry points are in `docs/`, `scripts/`, and `ci/`. Treat `build/` and local variants such as `build-kv260/` as generated output.

## Build, Test, and Development Commands
Use CMake, not the root `Makefile`.

- `cmake -B build -DCMAKE_BUILD_TYPE=Release`
  Configure a standard CPU build.
- `cmake --build build --config Release -j8`
  Build libraries, tools, and tests.
- `ctest --test-dir build --output-on-failure -L main`
  Run the main C/C++ test set.
- `./scripts/debug-test.sh test-tokenizer`
  Build and run a targeted test interactively.
- `bash ./ci/run.sh ./tmp/results ./tmp/mnt`
  Run the repo’s local CI flow; enable backends with env vars such as `GG_BUILD_CUDA=1`.
- `cd tools/server/webui && npm install && npm run test`
  Run Web UI checks; use `npm run dev` for local UI development.

## Coding Style & Naming Conventions
Follow `.editorconfig` and `.clang-format`: 4-space indentation, LF line endings, and a final newline; `Makefile`-style files use tabs. In C/C++, keep braces on the same line, prefer simple control flow over heavy template usage, and preserve cross-platform compatibility. Use `snake_case` for functions, variables, and types; C/C++ filenames are lowercase with dashes (`test-chat-template.cpp`), while Python files use lowercase with underscores.

## Testing Guidelines
Add or extend the nearest `tests/test-*.cpp` or `tests/*.sh` coverage for every behavior change. If you modify `ggml` operators or backend code, run `test-backend-ops`; for inference-path changes, also check `llama-bench` and `llama-perplexity`. Python changes should run with `pytest gguf-py/tests`. Web UI changes should run `npm run test:unit` or `npm run test:e2e` as appropriate.

## Commit & Pull Request Guidelines
Recent history favors module-first, imperative subjects such as `CUDA: fix bug in topk-moe softmax (#16711)`. `CONTRIBUTING.md` asks maintainers to squash using `<module> : <title> (#<issue>)`; matching that style in commits helps review. Keep PRs narrowly scoped, list the build/test commands you ran, link the issue or discussion, and include screenshots for `tools/server/webui` changes. Avoid mixing unrelated fixes or generated files in one PR.
