import math
import json
import argparse
import os
import onnx
from dataclasses import dataclass, asdict, replace
from typing import List, Dict, Optional, Any, Tuple

# ==========================================
# 1. 硬件与分块配置定义
# ==========================================
@dataclass
class HardwareConfig:
    """硬件配置参数"""
    spm_size_bytes: int
    acc_size_bytes: int
    
    dtype_input: int = 1     # int8
    dtype_acc: int = 4       # int32
    
    # 物理阵列参数
    sys_array_size: int = 32 
    measured_bandwidth_mbps: float = 1300.0 
    
    # 原子操作参数 (计算 32x32x32 卷积)
    measured_conv_us: float = 5.0 
    measured_gemm_us: float = 5.0
    instr_overhead_us: float = 1.54

    # GEMM 第二阶段硬件切分约束
    gemm_nm_split: int = 32
    gemm_k_split_limit: int = 2048

@dataclass
class TileResult:
    """单层分块评估结果"""
    t_oh: int
    t_ow: int
    t_ic: int
    t_oc: int
    latency_ms: float
    spm_util: float
    acc_util: float


@dataclass
class GemmTileResult:
    """单层 GEMM 分块评估结果"""
    t_n: int
    t_m: int
    t_k: int
    latency_ms: float
    spm_util: float
    acc_util: float
    stage1_k_tiles: int
    stage2_k_tiles: int
    dma_traffic_bytes: int


@dataclass
class LayerParams:
    """卷积层参数"""
    name: str
    H: int
    W: int
    IC: int
    OC: int
    K_h: int
    K_w: int
    S_h: int
    S_w: int
    D_h: int
    D_w: int
    P_top: int
    P_bottom: int
    P_left: int
    P_right: int

    @property
    def K_eff_h(self) -> int:
        return (self.K_h - 1) * self.D_h + 1

    @property
    def K_eff_w(self) -> int:
        return (self.K_w - 1) * self.D_w + 1

    @property
    def OH(self) -> int: 
        return (self.H + self.P_top + self.P_bottom - self.K_eff_h) // self.S_h + 1
        
    @property
    def OW(self) -> int: 
        return (self.W + self.P_left + self.P_right - self.K_eff_w) // self.S_w + 1


@dataclass
class GemmLayerParams:
    """GEMM / MatMul 层参数（统一为 [B, N, K] x [B, K, M] -> [B, N, M]）"""
    name: str
    op_type: str
    B: int
    N: int
    M: int
    K: int
    has_bias: bool = False
    trans_a: int = 0
    trans_b: int = 0

# ==========================================
# 2. ONNX 解析器
# ==========================================
class OnnxModelParser:
    """用于解析 ONNX 模型并提取卷积层信息的工具类"""

    @staticmethod
    def _decode_auto_pad(attr: Dict[str, Any]) -> str:
        if 'auto_pad' not in attr:
            return 'NOTSET'
        raw = attr['auto_pad'].s
        if isinstance(raw, bytes):
            return raw.decode('utf-8').upper()
        if isinstance(raw, str):
            return raw.upper()
        return 'NOTSET'

    @staticmethod
    def _compute_same_padding(
        in_size: int, stride: int, kernel: int, dilation: int, auto_pad: str
    ) -> tuple:
        out_size = math.ceil(in_size / stride)
        effective_kernel = (kernel - 1) * dilation + 1
        total_pad = max((out_size - 1) * stride + effective_kernel - in_size, 0)
        if auto_pad == 'SAME_LOWER':
            pad_before = (total_pad + 1) // 2
            pad_after = total_pad // 2
        else:
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        return pad_before, pad_after
    
    @staticmethod
    def _safe_dim_value(dim) -> int:
        if getattr(dim, 'dim_value', 0) > 0:
            return int(dim.dim_value)
        # 对符号/动态维度采用保守默认值 1，确保可建模
        return 1

    @staticmethod
    def _get_attr_int(node, attr_name: str, default: int = 0) -> int:
        for a in node.attribute:
            if a.name == attr_name:
                if hasattr(a, "i"):
                    return int(a.i)
        return default

    @staticmethod
    def _build_value_info(graph) -> Dict[str, Any]:
        value_info = {vi.name: vi for vi in graph.value_info}
        value_info.update({vi.name: vi for vi in graph.input})
        value_info.update({vi.name: vi for vi in graph.output})
        return value_info

    @staticmethod
    def _get_tensor_shape(
        tensor_name: str, value_info: Dict[str, Any], initializers: Dict[str, Any]
    ) -> Optional[List[int]]:
        if tensor_name in value_info:
            dims = value_info[tensor_name].type.tensor_type.shape.dim
            shape = [OnnxModelParser._safe_dim_value(d) for d in dims]
            return shape if shape else None
        if tensor_name in initializers:
            shape = [int(d) for d in initializers[tensor_name].dims]
            return shape if shape else None
        return None

    @staticmethod
    def parse(
        onnx_path: str,
        input_shape_override: Optional[List[int]] = None,
        op_type: str = "conv",
        dynamic_dim_default: Optional[int] = 1,
        dynamic_dim_map: Optional[Dict[str, int]] = None,
    ) -> List[Any]:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"未找到 ONNX 文件: {onnx_path}")

        print(f"[Parser] 正在加载模型: {onnx_path}...")
        model = onnx.load(onnx_path)
        graph = model.graph
        
        # 处理输入维度
        input_tensor = graph.input[0]
        orig_dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
        
        if input_shape_override:
            print(f"[Parser] 覆盖输入维度为: {input_shape_override}")
            if input_tensor.type.tensor_type.shape.dim:
                if len(input_shape_override) != len(orig_dims):
                    print(f"[Warning] 覆盖维度长度 {len(input_shape_override)} 与模型输入秩 {len(orig_dims)} 不一致")
                for i, dim_val in enumerate(input_shape_override):
                    if i < len(input_tensor.type.tensor_type.shape.dim):
                        input_tensor.type.tensor_type.shape.dim[i].dim_value = dim_val
        else:
            if any(d <= 0 for d in orig_dims):
                has_map = bool(dynamic_dim_map)
                has_default = dynamic_dim_default is not None and dynamic_dim_default > 0
                if (not has_map) and (not has_default):
                    raise ValueError(f"模型存在动态输入维度 {orig_dims}，请使用 --shape 参数指定静态维度。")
                print(
                    f"[Parser] 检测到动态输入维度 {orig_dims}，"
                    f"将进行静态化（默认值={dynamic_dim_default}，dim_map={dynamic_dim_map or {}}）。"
                )
            print(f"[Parser] 使用模型静态维度: {orig_dims}")

        # 对其余输入也做动态维度静态化，避免 shape_inference 在多输入模型上失败。
        can_fill_default = dynamic_dim_default is not None and dynamic_dim_default > 0
        if (dynamic_dim_map and len(dynamic_dim_map) > 0) or can_fill_default:
            map_replaced = 0
            default_replaced = 0
            for g_in in graph.input:
                for dim in g_in.type.tensor_type.shape.dim:
                    if dim.dim_value <= 0:
                        if dynamic_dim_map and dim.dim_param in dynamic_dim_map:
                            dim.dim_value = int(dynamic_dim_map[dim.dim_param])
                            map_replaced += 1
                        else:
                            if can_fill_default:
                                dim.dim_value = int(dynamic_dim_default)
                                default_replaced += 1
            total_replaced = map_replaced + default_replaced
            if total_replaced > 0:
                print(
                    f"[Parser] 已静态化动态维度共 {total_replaced} 个 "
                    f"(按 dim_map={map_replaced}, 按默认值={default_replaced})。"
                )

        # 形状推断
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"[Warning] 形状推断失败: {e}。后续维度可能不准确。")

        if op_type == "conv":
            return OnnxModelParser._extract_conv_layers(model.graph)
        if op_type == "gemm":
            return OnnxModelParser._extract_gemm_layers(model.graph)
        raise ValueError(f"不支持的 op_type: {op_type}")

    @staticmethod
    def _extract_conv_layers(graph) -> List[LayerParams]:
        """从计算图中提取 Conv 节点信息"""
        value_info = OnnxModelParser._build_value_info(graph)
        initializers = {init.name: init for init in graph.initializer}

        layers = []
        for node in graph.node:
            if node.op_type != 'Conv':
                continue
            if len(node.input) < 2:
                # Malformed or transformed Conv node; skip safely.
                continue

            attr = {a.name: a for a in node.attribute}
            k_h, k_w = None, None
            if 'kernel_shape' in attr and len(attr['kernel_shape'].ints) >= 2:
                kernel_shape = attr['kernel_shape'].ints
                k_h, k_w = kernel_shape[0], kernel_shape[1]

            strides = attr.get('strides').ints if 'strides' in attr else [1, 1]
            s_h = int(strides[0]) if len(strides) >= 1 else 1
            s_w = int(strides[1]) if len(strides) >= 2 else s_h

            dilations = attr.get('dilations').ints if 'dilations' in attr else [1, 1]
            d_h = int(dilations[0]) if len(dilations) >= 1 else 1
            d_w = int(dilations[1]) if len(dilations) >= 2 else d_h

            auto_pad = OnnxModelParser._decode_auto_pad(attr)

            pads = [0, 0, 0, 0]
            if 'pads' in attr and len(attr['pads'].ints) >= 4:
                pads = [int(x) for x in attr['pads'].ints[:4]]
            
            # 解析输入特征图 (IC, H, W)
            input_name = node.input[0]
            if input_name in value_info:
                input_shape = value_info[input_name].type.tensor_type.shape.dim
                if len(input_shape) >= 4:
                    ic = OnnxModelParser._safe_dim_value(input_shape[1])
                    h = OnnxModelParser._safe_dim_value(input_shape[2])
                    w = OnnxModelParser._safe_dim_value(input_shape[3])
                else:
                    continue
            else:
                continue

            # 解析权重 (OC, K_h, K_w)
            weight_name = node.input[1]
            if weight_name in initializers:
                weight_tensor = initializers[weight_name]
                if len(weight_tensor.dims) < 1:
                    continue
                oc = int(weight_tensor.dims[0])
                if (k_h is None or k_w is None) and len(weight_tensor.dims) >= 4:
                    k_h, k_w = int(weight_tensor.dims[2]), int(weight_tensor.dims[3])
            elif weight_name in value_info:
                 weight_shape = value_info[weight_name].type.tensor_type.shape.dim
                 if len(weight_shape) < 1:
                     continue
                 oc = OnnxModelParser._safe_dim_value(weight_shape[0])
                 if (k_h is None or k_w is None) and len(weight_shape) >= 4:
                     k_h = OnnxModelParser._safe_dim_value(weight_shape[2])
                     k_w = OnnxModelParser._safe_dim_value(weight_shape[3])
            else:
                continue

            if k_h is None or k_w is None or h <= 0 or w <= 0:
                continue

            if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
                p_top, p_bottom = OnnxModelParser._compute_same_padding(
                    h, s_h, k_h, d_h, auto_pad
                )
                p_left, p_right = OnnxModelParser._compute_same_padding(
                    w, s_w, k_w, d_w, auto_pad
                )
            elif auto_pad == 'VALID':
                p_top = p_bottom = p_left = p_right = 0
            else:
                p_top, p_left, p_bottom, p_right = pads

            layers.append(
                LayerParams(
                    name=node.name,
                    H=h,
                    W=w,
                    IC=ic,
                    OC=oc,
                    K_h=k_h,
                    K_w=k_w,
                    S_h=s_h,
                    S_w=s_w,
                    D_h=d_h,
                    D_w=d_w,
                    P_top=p_top,
                    P_bottom=p_bottom,
                    P_left=p_left,
                    P_right=p_right,
                )
            )

        return layers

    @staticmethod
    def _broadcast_batch_shape(a_batch: List[int], b_batch: List[int]) -> Optional[List[int]]:
        ra, rb = len(a_batch), len(b_batch)
        rank = max(ra, rb)
        a_pad = [1] * (rank - ra) + a_batch
        b_pad = [1] * (rank - rb) + b_batch
        out = []
        for da, db in zip(a_pad, b_pad):
            if da <= 0 or db <= 0:
                return None
            if da == db:
                out.append(da)
            elif da == 1:
                out.append(db)
            elif db == 1:
                out.append(da)
            else:
                return None
        return out

    @staticmethod
    def _extract_gemm_layers(graph) -> List[GemmLayerParams]:
        """从计算图中提取矩阵乘相关节点信息"""
        value_info = OnnxModelParser._build_value_info(graph)
        initializers = {init.name: init for init in graph.initializer}

        layers: List[GemmLayerParams] = []
        for idx, node in enumerate(graph.node):
            if node.op_type not in ("MatMul", "Gemm", "QLinearMatMul", "MatMulInteger"):
                continue

            # 不同算子的 A/B 输入位置
            a_idx = 0
            b_idx = 1
            if node.op_type == "QLinearMatMul":
                # QLinearMatMul: A, A_scale, A_zp, B, B_scale, B_zp, Y_scale, Y_zp
                a_idx = 0
                b_idx = 3

            if len(node.input) <= b_idx:
                continue

            a_shape = OnnxModelParser._get_tensor_shape(
                node.input[a_idx], value_info, initializers
            )
            b_shape = OnnxModelParser._get_tensor_shape(
                node.input[b_idx], value_info, initializers
            )
            if not a_shape or not b_shape:
                continue

            trans_a = 0
            trans_b = 0
            has_bias = False
            batch = 1

            if node.op_type == "Gemm":
                trans_a = OnnxModelParser._get_attr_int(node, "transA", 0)
                trans_b = OnnxModelParser._get_attr_int(node, "transB", 0)
                if len(a_shape) < 2 or len(b_shape) < 2:
                    continue

                n_dim = a_shape[-1] if trans_a else a_shape[-2]
                k_a = a_shape[-2] if trans_a else a_shape[-1]
                m_dim = b_shape[-2] if trans_b else b_shape[-1]
                k_b = b_shape[-1] if trans_b else b_shape[-2]
                has_bias = len(node.input) >= 3
            else:
                # MatMul: 统一到 [..., N, K] x [..., K, M]
                if len(a_shape) == 1:
                    a_shape = [1, a_shape[0]]
                if len(b_shape) == 1:
                    b_shape = [b_shape[0], 1]
                if len(a_shape) < 2 or len(b_shape) < 2:
                    continue

                n_dim = a_shape[-2]
                k_a = a_shape[-1]
                k_b = b_shape[-2]
                m_dim = b_shape[-1]

                batch_shape = OnnxModelParser._broadcast_batch_shape(
                    a_shape[:-2], b_shape[:-2]
                )
                if batch_shape is None:
                    continue
                batch = math.prod(batch_shape) if batch_shape else 1

            if min(n_dim, m_dim, k_a, k_b, batch) <= 0:
                continue
            if k_a != k_b:
                print(
                    f"[Warning] 跳过节点 {node.name or node.op_type}: "
                    f"K 维不匹配 (A:{k_a}, B:{k_b})"
                )
                continue

            layers.append(
                GemmLayerParams(
                    name=node.name or f"{node.op_type}_{idx}",
                    op_type=node.op_type,
                    B=batch,
                    N=n_dim,
                    M=m_dim,
                    K=k_a,
                    has_bias=has_bias,
                    trans_a=trans_a,
                    trans_b=trans_b,
                )
            )

        return layers

# ==========================================
# 3. 成本模型 (Cost Model)
# ==========================================
class CostModel:
    """评估特定分块策略在给定硬件下的延迟与内存利用率"""
    
    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    def _get_raw_input_tile_dim(self, t_oh: int, t_ow: int, layer: LayerParams) -> tuple:
        """计算生成 t_oh * t_ow 输出所需的输入尺寸"""
        t_ih = (t_oh - 1) * layer.S_h + layer.K_eff_h
        t_iw = (t_ow - 1) * layer.S_w + layer.K_eff_w
        return t_ih, t_iw

    def evaluate(self, layer: LayerParams, t_oh: int, t_ow: int, t_oc: int, t_ic: int) -> Optional[TileResult]:
        """评估分块配置，若超出内存限制则返回 None"""
        # 1. 空间需求计算
        raw_t_ih, raw_t_iw = self._get_raw_input_tile_dim(t_oh, t_ow, layer)
        
        size_ifm_tile = raw_t_ih * raw_t_iw * t_ic * self.hw.dtype_input
        size_wgt_tile = layer.K_h * layer.K_w * t_ic * t_oc * self.hw.dtype_input
        size_ofm_tile = t_oh * t_ow * t_oc * self.hw.dtype_input 
        
        total_spm_needed = size_ifm_tile + size_wgt_tile + size_ofm_tile
        size_acc_needed = t_oh * t_ow * t_oc * self.hw.dtype_acc

        # 内存溢出检查
        if total_spm_needed > self.hw.spm_size_bytes or size_acc_needed > self.hw.acc_size_bytes:
            return None

        # 2. 性能计算
        n_cout = math.ceil(layer.OC / t_oc)
        n_h    = math.ceil(layer.OH / t_oh)
        n_w    = math.ceil(layer.OW / t_ow)
        n_cin  = math.ceil(layer.IC / t_ic)

        total_global_tiles = n_cout * n_h * n_w * n_cin
        
        # DMA 数据量计算
        total_dma_ifm_bytes = total_global_tiles * size_ifm_tile
        total_dma_wgt_bytes = total_global_tiles * size_wgt_tile
        total_dma_bias_bytes = total_global_tiles * (t_oc * self.hw.dtype_acc)
        total_dma_ofm_bytes = (n_cout * n_h * n_w) * size_ofm_tile

        # 计算延迟
        num_spatial_micro_ops = math.ceil((t_oh * t_ow) / self.hw.sys_array_size)
        num_cout_micro_ops = math.ceil(t_oc / self.hw.sys_array_size)
        num_cin_micro_ops = math.ceil(t_ic / self.hw.sys_array_size)
        
        ops_per_conv_tile = num_cout_micro_ops * num_spatial_micro_ops * num_cin_micro_ops
        total_atomic_ops = total_global_tiles * ops_per_conv_tile
        
        total_traffic_bytes = total_dma_ifm_bytes + total_dma_wgt_bytes + total_dma_bias_bytes + total_dma_ofm_bytes
        num_dma_instructions = (total_global_tiles * 3) + (n_cout * n_h * n_w)

        lat_dma_us = (num_dma_instructions * self.hw.instr_overhead_us) + (total_traffic_bytes / self.hw.measured_bandwidth_mbps)
        lat_compute_us = total_atomic_ops * self.hw.measured_conv_us
        total_latency_ms = (lat_dma_us + lat_compute_us) / 1000.0

        return TileResult(
            t_oh=t_oh, t_ow=t_ow, t_ic=t_ic, t_oc=t_oc,
            latency_ms=total_latency_ms,
            spm_util=total_spm_needed / self.hw.spm_size_bytes,
            acc_util=size_acc_needed / self.hw.acc_size_bytes
        )


class GemmCostModel:
    """评估 GEMM 分块策略在给定硬件下的延迟与内存利用率"""

    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    @staticmethod
    def _align_tile(val: int, array_size: int = 32) -> int:
        if val < array_size:
            return max(1, val)
        return max(array_size, (val // array_size) * array_size)

    def calculate_auto_gemm_tile(self, m_dim: int, n_dim: int, k_dim: int) -> Tuple[int, int, int]:
        """
        对齐 C++ `calculateAutoGemmTile` 逻辑，返回 (t_m, t_n, t_k)。
        """
        input_dtype_bytes = self.hw.dtype_input
        output_dtype_bytes = self.hw.dtype_input
        acc_dtype_bytes = self.hw.dtype_acc

        m_aligned = self._align_tile(m_dim)
        n_aligned = self._align_tile(n_dim)

        # Step 1: 优先最大化 Tk
        min_tm = min(m_dim, self.hw.gemm_nm_split)
        min_tn = min(n_dim, self.hw.gemm_nm_split)
        base_out_spm = min_tm * min_tn * output_dtype_bytes

        max_tk_spm = 1
        if self.hw.spm_size_bytes > base_out_spm:
            denom = (min_tm + min_tn) * input_dtype_bytes
            if denom > 0:
                max_tk_spm = (self.hw.spm_size_bytes - base_out_spm) // denom
        t_k = max(1, min(k_dim, max_tk_spm))

        # Step 2: 固定 Tk 最大化 Tm
        max_tm_acc = self.hw.acc_size_bytes // max(1, (min_tn * acc_dtype_bytes))

        max_tm_spm = m_aligned
        spm_rem_for_m = self.hw.spm_size_bytes - min_tn * t_k * input_dtype_bytes
        if spm_rem_for_m > 0:
            denom = min_tn * output_dtype_bytes + t_k * input_dtype_bytes
            if denom > 0:
                max_tm_spm = spm_rem_for_m // denom

        t_m = min(m_aligned, max_tm_acc, max_tm_spm)
        t_m = self._align_tile(t_m)
        if t_m <= 0:
            return 0, 0, 0

        # Step 3: 固定 Tk/Tm 计算 Tn
        max_tn_acc = self.hw.acc_size_bytes // max(1, (t_m * acc_dtype_bytes))

        max_tn_spm = n_aligned
        spm_rem_for_n = self.hw.spm_size_bytes - t_k * t_m * input_dtype_bytes
        if spm_rem_for_n > 0:
            denom = t_m * output_dtype_bytes + t_k * input_dtype_bytes
            if denom > 0:
                max_tn_spm = spm_rem_for_n // denom

        t_n = min(n_aligned, max_tn_acc, max_tn_spm)
        t_n = self._align_tile(t_n)
        if t_n <= 0:
            return 0, 0, 0

        return t_m, t_n, t_k

    def evaluate(self, layer: GemmLayerParams, t_n: int, t_m: int, t_k: int) -> Optional[GemmTileResult]:
        if min(t_n, t_m, t_k) <= 0:
            return None

        # 1) 第一阶段分块的 SPM / ACC 约束
        size_a_tile = t_n * t_k * self.hw.dtype_input
        size_b_tile = t_k * t_m * self.hw.dtype_input
        size_out_tile = t_n * t_m * self.hw.dtype_input
        total_spm_needed = size_a_tile + size_b_tile + size_out_tile
        size_acc_needed = t_n * t_m * self.hw.dtype_acc

        if total_spm_needed > self.hw.spm_size_bytes or size_acc_needed > self.hw.acc_size_bytes:
            return None

        # 2) Stage1 分块次数（用于 DMA）
        n_n = math.ceil(layer.N / t_n)
        n_m = math.ceil(layer.M / t_m)
        n_k_stage1 = math.ceil(layer.K / t_k)
        output_tiles = layer.B * n_n * n_m
        total_stage1_tiles = output_tiles * n_k_stage1

        # 3) Stage2 约束（N/M=32，K<=2048）影响计算调用次数
        effective_k = min(t_k, self.hw.gemm_k_split_limit)
        n_k_stage2 = math.ceil(layer.K / max(1, effective_k))

        n_n_micro = math.ceil(t_n / self.hw.gemm_nm_split)
        n_m_micro = math.ceil(t_m / self.hw.gemm_nm_split)
        n_k_micro = math.ceil(effective_k / self.hw.gemm_nm_split)
        ops_per_compute_call = n_n_micro * n_m_micro * n_k_micro
        total_compute_calls = output_tiles * n_k_stage2
        total_atomic_ops = total_compute_calls * ops_per_compute_call

        # 4) DMA 估算
        total_dma_ifm_bytes = total_stage1_tiles * size_a_tile
        total_dma_wgt_bytes = total_stage1_tiles * size_b_tile
        total_dma_bias_bytes = output_tiles * (t_m * self.hw.dtype_acc) if layer.has_bias else 0
        total_dma_ofm_bytes = output_tiles * size_out_tile

        total_traffic_bytes = (
            total_dma_ifm_bytes
            + total_dma_wgt_bytes
            + total_dma_bias_bytes
            + total_dma_ofm_bytes
        )

        num_dma_instructions = total_stage1_tiles * 2 + output_tiles
        if layer.has_bias:
            num_dma_instructions += output_tiles

        lat_dma_us = (
            num_dma_instructions * self.hw.instr_overhead_us
            + (total_traffic_bytes / self.hw.measured_bandwidth_mbps)
        )
        lat_compute_us = total_atomic_ops * self.hw.measured_gemm_us
        total_latency_ms = (lat_dma_us + lat_compute_us) / 1000.0

        return GemmTileResult(
            t_n=t_n,
            t_m=t_m,
            t_k=t_k,
            latency_ms=total_latency_ms,
            spm_util=total_spm_needed / self.hw.spm_size_bytes,
            acc_util=size_acc_needed / self.hw.acc_size_bytes,
            stage1_k_tiles=n_k_stage1,
            stage2_k_tiles=n_k_stage2,
            dma_traffic_bytes=total_traffic_bytes,
        )

# ==========================================
# 4. 优化器 (Fixed Hardware Tiling Optimizer)
# ==========================================
class TileOptimizer:
    """在给定的硬件配置下，搜索最优的卷积分块策略"""
    
    def __init__(self, hw: HardwareConfig, layers: List[LayerParams]):
        self.hw = hw
        self.layers = layers
        self.cost_model = CostModel(hw)

    def _get_safe_range(self, limit: int, step: int, max_val: int = 64) -> List[int]:
        """生成安全的搜索空间范围"""
        end = min(limit, max_val)
        if end <= 0:
            return []
        if end < step:
            return [end]

        # Keep regular step-based candidates, and always include the exact
        # dimension boundary (e.g., 14) so full-tile options are evaluated.
        candidates = list(range(step, end + 1, step))
        if end not in candidates:
            candidates.append(end)
        return sorted(set(candidates))

    def search_best_tile(self, layer: LayerParams) -> Optional[TileResult]:
        """为单层搜索最佳分块"""
        min_lat = float('inf')
        best_res = None
        
        r_oh = self._get_safe_range(layer.OH, step=4, max_val=64) 
        r_ow = self._get_safe_range(layer.OW, step=4, max_val=64)
        
        start_ic = 32 if layer.IC >= 32 else layer.IC
        start_oc = 32 if layer.OC >= 32 else layer.OC
        r_ic = range(start_ic, min(layer.IC, 256) + 1, 32)
        r_oc = range(start_oc, min(layer.OC, 256) + 1, 32)

        for t_ic in r_ic:
            for t_oc in r_oc:
                for t_oh in r_oh:
                    for t_ow in r_ow:
                        res = self.cost_model.evaluate(layer, t_oh, t_ow, t_oc, t_ic)
                        if res and res.latency_ms < min_lat:
                            min_lat = res.latency_ms
                            best_res = res
                            
        return best_res

    def run_optimization(self) -> Optional[Dict[str, Any]]:
        """运行全网络层优化并汇总结果"""
        print(f"正在进行分块优化 (SPM: {self.hw.spm_size_bytes/1024:.0f} KB, ACC: {self.hw.acc_size_bytes/1024:.0f} KB)...")
        
        total_lat = 0.0
        layer_data = []
        
        for layer in self.layers:
            res = self.search_best_tile(layer)
            if not res:
                print(f"❌ 错误: 层 {layer.name} 在当前硬件限制下无法找到合适的分块(OOM)。")
                return None
            
            total_lat += res.latency_ms
            layer_data.append({
                'layer_name': layer.name,
                'input_shape': f"{layer.IC}x{layer.H}x{layer.W}",
                'output_shape': f"{layer.OC}x{layer.OH}x{layer.OW}",
                'tile_params': {
                    't_oh': res.t_oh, 't_ow': res.t_ow, 
                    't_ic': res.t_ic, 't_oc': res.t_oc
                },
                'latency_ms': res.latency_ms,
                'spm_util': res.spm_util,
                'acc_util': res.acc_util
            })
        
        return {
            'hw_config': {'spm_bytes': self.hw.spm_size_bytes, 'acc_bytes': self.hw.acc_size_bytes},
            'total_latency_ms': total_lat,
            'layers': layer_data
        }


class GemmTileOptimizer:
    """在给定硬件配置下，按当前编译器策略评估 GEMM 分块性能"""

    def __init__(self, hw: HardwareConfig, layers: List[GemmLayerParams]):
        self.hw = hw
        self.layers = layers
        self.cost_model = GemmCostModel(hw)

    def search_best_tile(self, layer: GemmLayerParams) -> Optional[GemmTileResult]:
        # 对齐当前 NPU 编译器第一阶段自动分块策略
        t_m, t_n, t_k = self.cost_model.calculate_auto_gemm_tile(
            layer.M, layer.N, layer.K
        )
        return self.cost_model.evaluate(layer, t_n, t_m, t_k)

    def run_optimization(self, verbose: bool = True) -> Optional[Dict[str, Any]]:
        if verbose:
            print(
                f"正在进行 GEMM 分块建模 "
                f"(SPM: {self.hw.spm_size_bytes/1024:.0f} KB, "
                f"ACC: {self.hw.acc_size_bytes/1024:.0f} KB)..."
            )

        total_lat = 0.0
        layer_data = []

        for layer in self.layers:
            res = self.search_best_tile(layer)
            if not res:
                if verbose:
                    print(f"❌ 错误: 层 {layer.name} 在当前硬件限制下无法找到合适分块(OOM)。")
                return None

            total_lat += res.latency_ms
            layer_data.append({
                'layer_name': layer.name,
                'op_type': layer.op_type,
                'problem_shape': {
                    'B': layer.B, 'N': layer.N, 'M': layer.M, 'K': layer.K
                },
                'tile_params': {
                    't_n': res.t_n, 't_m': res.t_m, 't_k': res.t_k
                },
                'stage2_constraints': {
                    'nm_split': self.hw.gemm_nm_split,
                    'k_split_limit': self.hw.gemm_k_split_limit,
                    'k_stage1_tiles': res.stage1_k_tiles,
                    'k_stage2_tiles': res.stage2_k_tiles,
                },
                'latency_ms': res.latency_ms,
                'spm_util': res.spm_util,
                'acc_util': res.acc_util,
                'dma_traffic_bytes': res.dma_traffic_bytes,
                'has_bias': layer.has_bias,
            })

        return {
            'hw_config': {
                'spm_bytes': self.hw.spm_size_bytes,
                'acc_bytes': self.hw.acc_size_bytes
            },
            'total_latency_ms': total_lat,
            'layers': layer_data,
            'meta': {
                'model': 'gemm',
                'strategy': (
                    'stage1_auto_tile(Tk->Tm->Tn), '
                    'stage2_split(N/M=32, K<=2048)'
                )
            }
        }


def _parse_kb_candidates(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    vals = []
    for x in raw.split(','):
        x = x.strip()
        if not x:
            continue
        try:
            v = int(x)
            if v > 0:
                vals.append(v)
        except ValueError:
            pass
    return sorted(set(vals))


def _parse_dynamic_dim_map(raw: Optional[str]) -> Dict[str, int]:
    if not raw:
        return {}
    result: Dict[str, int] = {}
    for part in raw.split(','):
        item = part.strip()
        if not item or '=' not in item:
            continue
        key, val = item.split('=', 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        try:
            num = int(val)
            if num > 0:
                result[key] = num
        except ValueError:
            pass
    return result


def _is_better_hw_candidate(a: Dict[str, Any], b: Optional[Dict[str, Any]]) -> bool:
    if b is None:
        return True
    eps = 1e-9
    if a['total_latency_ms'] + eps < b['total_latency_ms']:
        return True
    if abs(a['total_latency_ms'] - b['total_latency_ms']) <= eps:
        if a['total_sram_bytes'] < b['total_sram_bytes']:
            return True
        if (
            a['total_sram_bytes'] == b['total_sram_bytes']
            and abs(a['spm_bytes'] - a['acc_bytes']) < abs(b['spm_bytes'] - b['acc_bytes'])
        ):
            return True
    return False


def _build_pareto_frontier(valid_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_by_mem = sorted(valid_candidates, key=lambda x: (x['total_sram_bytes'], x['total_latency_ms']))
    frontier = []
    best_lat = float('inf')
    for c in sorted_by_mem:
        if c['total_latency_ms'] < best_lat:
            frontier.append(c)
            best_lat = c['total_latency_ms']
    return frontier


def _generate_sram_total_pairs(
    sram_total_kb: int, split_step_kb: int, min_part_kb: int = 16
) -> List[Tuple[int, int]]:
    if sram_total_kb <= 0 or split_step_kb <= 0:
        return []
    if sram_total_kb < min_part_kb * 2:
        return []

    pairs = set()
    for spm_kb in range(min_part_kb, sram_total_kb - min_part_kb + 1, split_step_kb):
        acc_kb = sram_total_kb - spm_kb
        if acc_kb >= min_part_kb:
            pairs.add((spm_kb, acc_kb))

    # 保证均分方案一定被评估
    balanced_spm = sram_total_kb // 2
    balanced_acc = sram_total_kb - balanced_spm
    if balanced_spm >= min_part_kb and balanced_acc >= min_part_kb:
        pairs.add((balanced_spm, balanced_acc))

    return sorted(pairs, key=lambda x: (x[0], x[1]))


def search_best_gemm_hw_config(
    layers: List[GemmLayerParams],
    hw_base: HardwareConfig,
    spm_candidates_kb: List[int],
    acc_candidates_kb: List[int],
    pairs_kb: Optional[List[Tuple[int, int]]] = None,
    topk: int = 10,
) -> Dict[str, Any]:
    all_candidates = []
    best_candidate = None
    best_detail = None

    candidate_pairs = []
    if pairs_kb:
        candidate_pairs = list(pairs_kb)
    else:
        candidate_pairs = [
            (spm_kb, acc_kb)
            for spm_kb in spm_candidates_kb
            for acc_kb in acc_candidates_kb
        ]

    total_pairs = len(candidate_pairs)
    print(f"正在进行 GEMM SPM/ACC 组合搜索，共 {total_pairs} 组候选...")

    for spm_kb, acc_kb in candidate_pairs:
        hw = replace(
            hw_base,
            spm_size_bytes=spm_kb * 1024,
            acc_size_bytes=acc_kb * 1024,
        )
        optimizer = GemmTileOptimizer(hw, layers)
        result = optimizer.run_optimization(verbose=False)

        candidate = {
            'spm_bytes': hw.spm_size_bytes,
            'acc_bytes': hw.acc_size_bytes,
            'total_sram_bytes': hw.spm_size_bytes + hw.acc_size_bytes,
            'spm_kb': spm_kb,
            'acc_kb': acc_kb,
            'valid': result is not None,
        }

        if result is not None:
            candidate['total_latency_ms'] = result['total_latency_ms']
            if _is_better_hw_candidate(candidate, best_candidate):
                best_candidate = candidate
                best_detail = result
        else:
            candidate['total_latency_ms'] = None

        all_candidates.append(candidate)

    valid_candidates = [c for c in all_candidates if c['valid']]
    valid_candidates_sorted = sorted(
        valid_candidates,
        key=lambda x: (x['total_latency_ms'], x['total_sram_bytes'], abs(x['spm_bytes'] - x['acc_bytes']))
    )
    pareto = _build_pareto_frontier(valid_candidates)

    summary = {
        'total_candidates': len(all_candidates),
        'valid_candidates': len(valid_candidates),
        'invalid_candidates': len(all_candidates) - len(valid_candidates),
    }

    result: Dict[str, Any] = {
        'mode': 'gemm_hw_search',
        'search_space': {
            'spm_candidates_kb': spm_candidates_kb,
            'acc_candidates_kb': acc_candidates_kb,
        },
        'summary': summary,
        'top_candidates': valid_candidates_sorted[:max(1, topk)],
        'pareto_frontier': pareto,
    }

    if best_candidate is not None and best_detail is not None:
        result['best_config'] = best_candidate
        result['best_config_detail'] = best_detail
    else:
        result['best_config'] = None
        result['best_config_detail'] = None

    if pairs_kb:
        result['search_space']['pairs_kb'] = pairs_kb
        totals = sorted(set(spm + acc for spm, acc in pairs_kb))
        if len(totals) == 1:
            result['search_space']['sram_total_kb'] = totals[0]

    return result

# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiling Optimizer for ONNX Models (Fixed Hardware)")
    parser.add_argument('-i', '--input', type=str, required=True, help='输入 ONNX 模型文件路径')
    parser.add_argument('-o', '--output', type=str, required=False, help='输出 JSON 文件路径')
    parser.add_argument('--op', type=str, choices=['conv', 'gemm'], default='conv', help='建模算子类型: conv 或 gemm')
    parser.add_argument('--spm', type=int, required=False, default=None, help='SPM 容量大小 (单位: KB)')
    parser.add_argument('--acc', type=int, required=False, default=None, help='ACC 容量大小 (单位: KB)')
    parser.add_argument('--sram-total', type=int, default=None, help='仅 GEMM 模式: 片上存储总量 (KB)，自动搜索 SPM/ACC 划分')
    parser.add_argument('--split-step-kb', type=int, default=16, help='仅 GEMM 模式且使用 --sram-total 时，SPM/ACC 划分步长 (KB)')
    parser.add_argument('--dynamic-dim-default', type=int, default=1, help='模型含动态维度时用于静态化的默认值')
    parser.add_argument('--dynamic-dim-map', type=str, default=None, help='按符号维度名指定静态值，如 batch_size=1,sequence_length=128')
    parser.add_argument('--shape', type=str, default=None, help='覆盖输入形状, 格式: N,C,H,W (例如 1,3,224,224)')
    parser.add_argument('--spm-candidates', type=str, default=None, help='仅 GEMM 模式: SPM 搜索候选 (KB, 逗号分隔)，如 64,96,128,160')
    parser.add_argument('--acc-candidates', type=str, default=None, help='仅 GEMM 模式: ACC 搜索候选 (KB, 逗号分隔)，如 64,96,128,160')
    parser.add_argument('--topk', type=int, default=10, help='仅 GEMM 搜索模式: 输出前 K 个候选')
    
    args = parser.parse_args()

    # 处理输入参数
    if args.op == 'conv':
        if args.spm is None or args.acc is None:
            print("❌ Conv 模式下必须显式指定 --spm 和 --acc。")
            exit(1)
    else:
        # GEMM 模式：支持只给总量后自动搜索划分
        if args.sram_total is not None and args.sram_total <= 0:
            print("❌ --sram-total 必须为正整数 (KB)。")
            exit(1)
        if args.split_step_kb <= 0:
            print("❌ --split-step-kb 必须为正整数。")
            exit(1)

        if args.spm is None and args.acc is None:
            if args.sram_total is None:
                print("❌ GEMM 模式请提供 --spm/--acc，或直接提供 --sram-total。")
                exit(1)
            # 默认均分作为基线配置
            args.spm = args.sram_total // 2
            args.acc = args.sram_total - args.spm
        elif args.spm is None:
            if args.sram_total is None:
                print("❌ 指定了 --acc 但缺少 --spm（且未提供 --sram-total）。")
                exit(1)
            args.spm = args.sram_total - args.acc
        elif args.acc is None:
            if args.sram_total is None:
                print("❌ 指定了 --spm 但缺少 --acc（且未提供 --sram-total）。")
                exit(1)
            args.acc = args.sram_total - args.spm

        if args.spm <= 0 or args.acc <= 0:
            print("❌ 计算得到的 --spm/--acc 非法（<=0），请检查参数。")
            exit(1)

    spm_bytes = args.spm * 1024
    acc_bytes = args.acc * 1024
    output_path = args.output or f"{os.path.splitext(os.path.basename(args.input))[0]}_optimized.json"

    shape_override = None
    dynamic_dim_map = _parse_dynamic_dim_map(args.dynamic_dim_map)
    if args.shape:
        try:
            shape_override = [int(x) for x in args.shape.split(',')]
        except ValueError:
            print("❌ 错误: --shape 参数格式无效。")
            exit(1)

    # 1. 解析 ONNX
    try:
        layers = OnnxModelParser.parse(
            args.input,
            input_shape_override=shape_override,
            op_type=args.op,
            dynamic_dim_default=args.dynamic_dim_default,
            dynamic_dim_map=dynamic_dim_map,
        )
    except Exception as e:
        print(f"❌ 解析 ONNX 模型失败: {e}")
        exit(1)

    def _write_json(data: Dict[str, Any]) -> bool:
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"✅ 结果已成功导出至 {output_path}")
            return True
        except Exception as e:
            print(f"❌ 导出 JSON 失败: {e}")
            return False
    
    hw_config = HardwareConfig(spm_size_bytes=spm_bytes, acc_size_bytes=acc_bytes)

    if args.op == 'conv':
        if not layers:
            empty_result = {
                'hw_config': {'spm_bytes': spm_bytes, 'acc_bytes': acc_bytes},
                'total_latency_ms': 0.0,
                'layers': [],
                'meta': {
                    'model_path': args.input,
                    'message': '未检测到有效的卷积(Conv)操作，输出空分块配置。'
                }
            }
            print("ℹ️ 未在模型中检测到有效的卷积 (Conv) 操作，正在输出空分块配置 JSON。")
            _write_json(empty_result)
            exit(0)

        print(f"\n模型: {args.input} (共检测到 {len(layers)} 个卷积层)")
        print(f"输出路径: {output_path}")

        optimizer = TileOptimizer(hw_config, layers)
        result = optimizer.run_optimization()
        if result:
            print("\n" + "=" * 60)
            print("✅ 成功找到最佳分块配置")
            print(f"  总延迟评估: {result['total_latency_ms']:.2f} ms")
            print("=" * 60)
            _write_json(result)
        else:
            print("\n❌ 优化失败：硬件资源限制过严，部分层发生 OOM。")
        exit(0)

    # GEMM / MatMul 路径
    if not layers:
        empty_result = {
            'hw_config': {'spm_bytes': spm_bytes, 'acc_bytes': acc_bytes},
            'total_latency_ms': 0.0,
            'layers': [],
            'meta': {
                'model_path': args.input,
                'message': '未检测到有效的 GEMM/MatMul 操作，输出空分块配置。'
            }
        }
        print("ℹ️ 未在模型中检测到有效的 GEMM/MatMul 操作，正在输出空分块配置 JSON。")
        _write_json(empty_result)
        exit(0)

    print(f"\n模型: {args.input} (共检测到 {len(layers)} 个 GEMM/MatMul 层)")
    print(f"输出路径: {output_path}")

    spm_candidates = _parse_kb_candidates(args.spm_candidates)
    acc_candidates = _parse_kb_candidates(args.acc_candidates)
    pair_candidates = None

    if args.sram_total is not None:
        pair_candidates = _generate_sram_total_pairs(
            sram_total_kb=args.sram_total,
            split_step_kb=args.split_step_kb,
            min_part_kb=max(16, args.split_step_kb),
        )
        if not pair_candidates:
            print("❌ 根据给定的总量/步长未生成有效的 SPM/ACC 划分候选。")
            exit(1)
        if (args.spm, args.acc) not in pair_candidates and (args.spm + args.acc == args.sram_total):
            pair_candidates.append((args.spm, args.acc))
            pair_candidates = sorted(set(pair_candidates))

    do_hw_search = bool(spm_candidates or acc_candidates or pair_candidates)

    if do_hw_search:
        if pair_candidates is None:
            if args.spm not in spm_candidates:
                spm_candidates.append(args.spm)
            if args.acc not in acc_candidates:
                acc_candidates.append(args.acc)
            spm_candidates = sorted(set(spm_candidates))
            acc_candidates = sorted(set(acc_candidates))
        else:
            spm_candidates = sorted(set([x[0] for x in pair_candidates]))
            acc_candidates = sorted(set([x[1] for x in pair_candidates]))

        search_result = search_best_gemm_hw_config(
            layers=layers,
            hw_base=hw_config,
            spm_candidates_kb=spm_candidates,
            acc_candidates_kb=acc_candidates,
            pairs_kb=pair_candidates,
            topk=args.topk,
        )

        if search_result.get('best_config') is not None:
            best = search_result['best_config']
            print("\n" + "=" * 60)
            print("✅ GEMM 硬件搜索完成")
            print(
                f"  最优配置: SPM={best['spm_kb']}KB, ACC={best['acc_kb']}KB, "
                f"Latency={best['total_latency_ms']:.2f} ms"
            )
            print("=" * 60)
        else:
            print("\n❌ GEMM 硬件搜索失败：所有候选配置均 OOM。")

        _write_json(search_result)
    else:
        optimizer = GemmTileOptimizer(hw_config, layers)
        result = optimizer.run_optimization()

        if result:
            print("\n" + "=" * 60)
            print("✅ GEMM 分块建模完成")
            print(f"  总延迟评估: {result['total_latency_ms']:.2f} ms")
            print("=" * 60)
            _write_json(result)
        else:
            print("\n❌ 优化失败：硬件资源限制过严，部分 GEMM 层发生 OOM。")
