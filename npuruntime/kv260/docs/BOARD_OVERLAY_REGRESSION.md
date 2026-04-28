# KV260 Overlay 回归测试使用说明

本文档面向只在 KV260 板子上运行测试的使用者。你不需要访问开发机仓库，只需要板子上已经准备好以下目录：

```text
/home/ubuntu/kv260-regression
```

## 1. 前置条件

板子需要满足：

- 使用 KV260 Ubuntu/PetaLinux 环境，并带有 `xmutil`。
- 要测试的 overlay app 已经由你安装到 `xmutil` 可加载的位置。
- 默认测试的 app 名字是：`double_dma_overlayapp`。
- 当前用户可以执行 `sudo`，或者已经配置无密码 sudo。
- `/home/ubuntu/kv260-regression` 下存在：
  - `scripts/run_regression.sh`
  - `driver/npu_kv260.ko`
  - `bin/kv260_*` 测试程序

先确认 app 能被 `xmutil` 看到：

```bash
xmutil listapps
```

如果你的 app 名不是 `double_dma_overlayapp`，运行测试时需要用 `--app-name` 指定。

## 2. 一键运行

进入 regression 目录：

```bash
cd /home/ubuntu/kv260-regression
```

如果板子需要 sudo 密码，默认密码为123456，先设置环境变量。下面只是示例，实际密码由你的板端环境决定：

```bash
export KV260_SUDO_PASSWORD='<board-sudo-password>'
```

运行完整回归：

```bash
./scripts/run_regression.sh
```

指定 app 名运行：

```bash
./scripts/run_regression.sh --app-name my_overlay_app
```

快速 smoke profile：

```bash
./scripts/run_regression.sh --profile fast
```

失败后继续跑剩余 case：

```bash
./scripts/run_regression.sh --continue-on-fail
```

## 3. 测试行为

每个测试 case 开始前都会重新执行：

```bash
sudo rmmod npu_kv260 || true
sudo xmutil unloadapp || true
sudo xmutil loadapp <app-name>
sudo insmod /home/ubuntu/kv260-regression/driver/npu_kv260.ko
sudo chgrp <current-group> /dev/npu_kv260
sudo chmod 660 /dev/npu_kv260
```

这样做是为了隔离测试，避免上一个 case 的硬件或驱动状态污染下一个 case。

## 4. 测试内容

`fast` profile 包含：

- `readiness`：检查 `xmutil`、platform device、驱动模块和 `/dev/npu_kv260`。
- `smoke_1m`：驱动和 1 MiB CMA mmap smoke test。
- `runtime_init`：runtime 初始化和小块内存分配。
- `dma_loopback`：基础 DMA loopback。

`full` profile 在 `fast` 基础上增加：

- `smoke_256m`：256 MiB CMA 分配和 mmap 验证。
- `mvin_problem_case`：MVIN/MVOUT 重复问题 case。
- `dma_2d_submatrix`：DRAM stride 大于子矩阵宽度、SPM stride 等于子矩阵宽度的 2D DMA 子矩阵搬运。
- `dma_acc_int32_fp32`：ACC int32/fp32 读回路径。
- `double_mvin_async`：双 DMA MVIN 异步路径。
- `gemm_basic`：基础 GEMM replay。
- `gemm_double_dma`：双 DMA GEMM replay。
- `mmproj_asym_w8a8`：MMProj asymmetric W8A8 layer replay。

## 5. 结果目录

每次运行都会生成一个目录：

```text
/home/ubuntu/kv260-regression/runs/<timestamp>/
```

关键文件：

- `summary.tsv`：每个 case 的 PASS/FAIL、退出码、耗时和日志路径。
- `run_meta.json`：app 名、profile、系统信息和运行参数。
- `cases/*.log`：每个 case 的完整日志。
- `cases/*.readiness.txt`：每个 case 重新加载 overlay 后的板端状态。
- `cases/*.dmesg_tail.txt`：失败 case 的内核日志尾部。

查看 summary：

```bash
cat runs/*/summary.tsv | tail -n 20
```

如果最新一次运行失败，优先查看：

```bash
latest=$(ls -dt runs/* | head -n 1)
cat "$latest/summary.tsv"
ls "$latest/cases"
```

## 6. PASS/FAIL 判定

完整回归退出码为 `0` 表示全部通过。任一 case 失败时，脚本默认立即停止并返回非零退出码。

如果使用了 `--continue-on-fail`，脚本会继续执行后续 case，但最终只要有任一失败，整体仍返回非零。

## 7. 常见问题

### `xmutil is not available`

板端环境没有 `xmutil`，或当前 PATH 找不到。确认系统镜像是否支持 xmutil。

### `xmutil loadapp <app> failed`

通常表示 app 名写错，或 app 没有正确安装到 `/lib/firmware/xilinx/<app-name>`。

先运行：

```bash
xmutil listapps
```

确认 app 名后再运行：

```bash
./scripts/run_regression.sh --app-name <app-name>
```

### `/dev/npu_kv260 is missing`

可能原因：

- overlay 没有创建设备树 platform device。
- `npu_kv260.ko` 和当前内核不匹配。
- 驱动加载失败。

查看最新失败日志：

```bash
latest=$(ls -dt runs/* | head -n 1)
cat "$latest/cases/readiness.log"
cat "$latest/cases/readiness.dmesg_tail.txt"
```

### `Missing sudo password in env var`

当前用户没有无密码 sudo。设置：

```bash
export KV260_SUDO_PASSWORD='<board-sudo-password>'
```

然后重新运行。

### 测试卡住

默认每个 case 最多运行 300 秒。如果需要更长时间：

```bash
./scripts/run_regression.sh --timeout 900
```

## 8. 交付日志给维护者

遇到失败时，把最新 run 目录打包：

```bash
latest=$(ls -dt /home/ubuntu/kv260-regression/runs/* | head -n 1)
tar -C "$(dirname "$latest")" -czf /tmp/kv260-regression-failure.tar.gz "$(basename "$latest")"
ls -lh /tmp/kv260-regression-failure.tar.gz
```

把 `/tmp/kv260-regression-failure.tar.gz` 发给维护者即可。
