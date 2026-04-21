# 主机环境说明

`VersaVLM/` 是一个独立交付目录。

主办方侧只需要：

- `ssh`
- `scp`
- `python3`
- 一个完整未缺失的 `VersaVLM/` 目录

## 主机执行原则

- 所有命令都以 `VersaVLM/` 为根目录执行
- 不依赖原始开发仓库
- 不依赖外部模型目录
- 不依赖外部评测脚本目录

## 自检脚本

```bash
bash scripts/validate_bundle.sh
```

该脚本会校验：

- 模型
- mmproj
- overlay
- 驱动
- runtime libs
- 100 样本 json
- throughput 必要图片
- 测试程序
- 中英文文档
