# 主机环境说明

## 角色分离

- `AICAS/`：开发与历史实验区
- `AICAS2026/V3/`：正式提交与复现区

不要把开发期临时输出直接混入 `V3`。

## V3 中应包含的内容

- 正式脚本
- 正式文档
- 自包含 payload
- 正式结果
- manifest 与 checksum
- 源码状态快照

## 构建说明

默认 `prepare_v3_bundle.sh` 读取：

- SDK env：`/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux`
- build 目录：`build-kv260-npu-current`

如果已经有可用板端二进制，推荐：

```bash
bash AICAS2026/V3/scripts/prepare_v3_bundle.sh --skip-build
```

## 远端连接默认值

- host: `192.168.0.10`
- user: `ubuntu`
- remote root: `/home/ubuntu/aicas`
- sudo password: `123456`

主办方如果环境不同，应在运行脚本时显式覆盖，不要改脚本默认值。
