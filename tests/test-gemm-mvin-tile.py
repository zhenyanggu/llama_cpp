#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "gemm_mvin_tile.py"
SPEC = importlib.util.spec_from_file_location("gemm_mvin_tile", MODULE_PATH)
gemm_mvin_tile = importlib.util.module_from_spec(SPEC)
import sys
assert SPEC.loader is not None
sys.modules["gemm_mvin_tile"] = gemm_mvin_tile
SPEC.loader.exec_module(gemm_mvin_tile)


LayerProfile = gemm_mvin_tile.LayerProfile
LayerGroup = gemm_mvin_tile.LayerGroup


class GemmMvinTileTest(unittest.TestCase):
    def _make_group(self, m=768, n=1024, k=768, count=1, stage2_k_block=4096):
        layers = [
            LayerProfile(
                layer_id=i,
                semantic_op="attn_q",
                root_op_name="ADD",
                m=m,
                n=n,
                k=k,
                spm_bytes=512 * 1024,
                acc_bytes=512 * 1024,
                stage2_k_block=stage2_k_block,
                first_stage_tm=640,
                first_stage_tn=32,
                first_stage_tk=768,
                exec_tile_count=768,
                activation_bytes=18_874_368,
                weight_bytes=18_874_368,
                output_bytes=3_145_728,
                activation_calls=768,
                weight_calls=768,
                raw_node={},
            )
            for i in range(count)
        ]
        return LayerGroup(
            group_key="shape:test",
            semantic_op="attn_q",
            m=m,
            n=n,
            k=k,
            spm_bytes=512 * 1024,
            acc_bytes=512 * 1024,
            stage2_k_block=stage2_k_block,
            count=count,
            members=layers,
        )

    def test_split_blocks_keep_tail(self):
        self.assertEqual(gemm_mvin_tile._split_blocks(70, 32), [32, 32, 6])
        self.assertEqual(gemm_mvin_tile._macro_micro_blocks(70, 64, 32), [32, 32, 6])

    def test_candidate_respects_memory_constraints(self):
        group = self._make_group(k=8192, stage2_k_block=8192)
        result = gemm_mvin_tile.evaluate_candidate(
            group=group,
            tm=768,
            tn=1024,
            tk=8192,
            schedule="reuse_weight",
            split_same_matrix=False,
            dma_setup_us=1.54,
            dma_bw_bytes_per_us=1300.0,
            guard_bytes=4 * 1024,
        )
        self.assertIsNone(result)

    def test_reuse_weight_reduces_weight_loads(self):
        group = self._make_group()
        candidate = gemm_mvin_tile.evaluate_candidate(
            group=group,
            tm=640,
            tn=32,
            tk=768,
            schedule="reuse_weight",
            split_same_matrix=False,
            dma_setup_us=1.54,
            dma_bw_bytes_per_us=1300.0,
            guard_bytes=4 * 1024,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertLess(candidate.weight_loads, group.members[0].weight_calls)
        self.assertLess(candidate.total_bytes, group.members[0].activation_bytes + group.members[0].weight_bytes)

    def test_split_same_matrix_improves_single_matrix_events(self):
        group = self._make_group()
        no_split = gemm_mvin_tile.evaluate_candidate(
            group=group,
            tm=640,
            tn=32,
            tk=768,
            schedule="reuse_weight",
            split_same_matrix=False,
            dma_setup_us=1.54,
            dma_bw_bytes_per_us=1300.0,
            guard_bytes=4 * 1024,
        )
        split_mode = gemm_mvin_tile.evaluate_candidate(
            group=group,
            tm=640,
            tn=32,
            tk=768,
            schedule="reuse_weight",
            split_same_matrix=True,
            dma_setup_us=1.54,
            dma_bw_bytes_per_us=1300.0,
            guard_bytes=4 * 1024,
        )
        self.assertIsNotNone(no_split)
        self.assertIsNotNone(split_mode)
        assert no_split is not None and split_mode is not None
        self.assertLess(split_mode.est_mvin_us, no_split.est_mvin_us)


if __name__ == "__main__":
    unittest.main()
