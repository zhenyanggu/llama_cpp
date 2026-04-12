# KV260 Ubuntu NPU Stack

This directory contains the KV260-specific NPU driver and runtime for the
official Ubuntu flow. It does not depend on a reserved-memory node. The driver
allocates contiguous memory through the Linux DMA/CMA API and returns the DMA
address to user space.

## What Changed From zcu102

- zcu102 maps a fixed DDR physical window and returns a hard-coded DDR base.
- KV260 allocates a per-open CMA buffer with `dma_alloc_coherent()`.
- KV260 exposes the buffer through `mmap(offset=0)` and the register window
  through `mmap(offset=0x10000000)`.
- KV260 keeps the existing register and IRQ-mode behavior, but reset and clock
  controls are optional because the `xmutil` app DT overlay may not provide a
  reset phandle on the NPU node.
- The NPU DMA address fields are 32-bit, so the driver sets a 32-bit DMA mask
  and rejects CMA buffers outside that range.

## Layout

- `driver/npu_kv260.c`: out-of-tree Linux kernel module.
- `driver/npu_kv260_uapi.h`: ioctl structs, commands, mmap offsets.
- `driver/Makefile`: kernel module build.
- `runtime/npu_runtime.cpp`: KV260 runtime copied from the zcu102 API surface
  and changed to request/mmap a CMA heap.
- `runtime/smoke_test.c`: low-level driver smoke test.
- `runtime/runtime_init_test.cpp`: runtime init and allocator smoke test.
- `runtime/Makefile`: runtime library and test builds.

## Build

On a KV260 Ubuntu image with build tools installed:

```sh
make -C kv260/driver
make -C kv260/runtime
```

If the Ubuntu rootfs does not have build tools, cross-build from a host with a
KV260/PetaLinux aarch64 SDK and the target Ubuntu kernel headers. The tested
target was `6.8.0-1015-xilinx`; the board had headers in
`/usr/src/linux-headers-6.8.0-1015-xilinx` and common scripts in
`/usr/src/linux-xilinx-headers-6.8.0-1015`.

The module was verified by copying those header trees to `/tmp`, rebuilding only
the host-side `scripts/basic/fixdep` and `scripts/mod/modpost` tools as x86-64
binaries, then running:

```sh
source /path/to/environment-setup-cortexa72-cortexa53-amd-linux
make -C /tmp/kv260-linux-headers-6.8.0-1015-xilinx \
  M=$PWD/kv260/driver \
  ARCH=arm64 CROSS_COMPILE=aarch64-amd-linux- \
  CC=aarch64-amd-linux-gcc LD=aarch64-amd-linux-ld AR=aarch64-amd-linux-ar \
  HOSTCC=gcc modules
```

For runtime cross-build:

```sh
source /path/to/environment-setup-cortexa72-cortexa53-amd-linux
make -C kv260/runtime clean all CXX="$CXX" CC="$CC" AR=aarch64-amd-linux-ar
```

## Load And Test

Load the dynamic PL app first:

```sh
sudo xmutil unloadapp
sudo xmutil loadapp mynpu
sudo xmutil listapps
```

The tested `mynpu` DT overlay created this platform device:

```text
a0000000.T_NPU_FPGA
compatible = "xlnx,T-NPU-FPGA-1.0"
reg = 0xa0000000 size 0x400
irq = GIC SPI 0x59
```

Load the driver:

```sh
sudo insmod npu_kv260.ko
dmesg | tail
ls -l /dev/npu_kv260
```

If no DT platform device exists, use the fallback module parameters:

```sh
sudo insmod npu_kv260.ko reg_base=0xa0000000 reg_size=0x400 irq=55
```

Run the smoke tests:

```sh
sudo ./kv260_npu_smoke_test 1M
sudo ./kv260_npu_smoke_test 256M
sudo env NPU_CMA_SIZE=256M ./kv260_runtime_init_test
sudo env NPU_CMA_SIZE=16M ./kv260_dma_loopback_test
```

Expected output includes:

```text
cma_mmap_rw=ok
kv260_npu_smoke_test=ok
kv260_runtime_init_test=ok
kv260_dma_loopback_test=ok
```

## Runtime Notes

- The runtime opens `/dev/npu_kv260`.
- `NPU_CMA_SIZE` controls the initial heap request; default is `256M`.
- `npu_mem_alloc()` sub-allocates from that single CMA heap.
- The device node is root-owned by default on Ubuntu, so use `sudo` or add a
  local udev rule if non-root access is desired.
- Full NPU task tests still depend on the loaded PL image and valid register
  semantics; the included smoke tests only validate driver probing, CMA
  allocation, mmap, DMA address reporting, and basic register access.
