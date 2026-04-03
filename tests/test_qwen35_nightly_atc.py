import datetime as dt
import unittest

import qwen35_nightly_atc as nightly


class Qwen35NightlyAtcTest(unittest.TestCase):
    def test_compute_deadline_uses_same_day_when_before_target(self):
        now = dt.datetime(2026, 3, 21, 1, 50, 0)

        deadline = nightly.compute_deadline(now, "09:00")

        self.assertEqual(deadline, dt.datetime(2026, 3, 21, 9, 0, 0))

    def test_compute_deadline_rolls_to_next_day_when_after_target(self):
        now = dt.datetime(2026, 3, 21, 10, 5, 0)

        deadline = nightly.compute_deadline(now, "09:00")

        self.assertEqual(deadline, dt.datetime(2026, 3, 22, 9, 0, 0))

    def test_build_case_matrix_expands_seq_precision_and_variant(self):
        cases = nightly.build_case_matrix(
            seq_lens=[64, 128],
            precision_modes=["allow_fp32_to_fp16", "must_keep_origin_dtype"],
            onnx_variants=["raw", "patched"],
        )

        self.assertEqual(len(cases), 8)
        self.assertEqual(cases[0].case_id, "seq64_allow_fp32_to_fp16_raw")
        self.assertEqual(cases[-1].case_id, "seq128_must_keep_origin_dtype_patched")

    def test_classify_atc_failure_prefers_linear_attention_root_cause(self):
        log_text = """
        [ERROR] GE: Operator [Mul] in node [/model/layers.0/linear_attn/Mul] is not supported
        [ERROR] Graph engine compile failed
        """

        failure_kind = nightly.classify_atc_failure(log_text)

        self.assertEqual(failure_kind, "linear_attention_unsupported")

    def test_classify_atc_failure_detects_rmsnorm_family(self):
        log_text = """
        [ERROR] op type aclnnRmsNorm is not registered
        [ERROR] Build graph failed because RmsNorm is unsupported
        """

        failure_kind = nightly.classify_atc_failure(log_text)

        self.assertEqual(failure_kind, "rmsnorm_unsupported")

    def test_should_launch_new_case_respects_guard_window(self):
        now = dt.datetime(2026, 3, 21, 8, 45, 0)
        deadline = dt.datetime(2026, 3, 21, 9, 0, 0)

        self.assertFalse(nightly.should_launch_new_case(now, deadline, guard_minutes=20))
        self.assertTrue(nightly.should_launch_new_case(now, deadline, guard_minutes=10))

    def test_render_final_report_contains_key_conclusions(self):
        case = nightly.NightlyCase(
            seq_len=64,
            precision_mode="allow_fp32_to_fp16",
            onnx_variant="raw",
        )
        summary = nightly.NightlySummary(
            run_id="20260321_015000",
            deadline=dt.datetime(2026, 3, 21, 9, 0, 0),
            finished_at=dt.datetime(2026, 3, 21, 8, 30, 0),
            toolchain_status="ready",
            board_model_path="/home/HwHiAiUser/ICT/models/qwen3.5-0.8b",
            board_onnx_path="/home/HwHiAiUser/ICT/qwen35_0p8b_onnx/model.onnx",
            model_has_linear_attention=True,
            discovered_rms_norm=True,
            case_results=[
                nightly.CaseResult(
                    case=case,
                    status="failed",
                    failure_kind="linear_attention_unsupported",
                    om_path=None,
                    started_at="2026-03-21 02:00:00",
                    finished_at="2026-03-21 03:00:00",
                )
            ],
            board_smoke_status="not_run",
        )

        report = nightly.render_final_report(summary)

        self.assertIn("# Qwen3.5-0.8B 夜间 ATC 测试报告", report)
        self.assertIn("toolchain_status: ready", report)
        self.assertIn("model_has_linear_attention: yes", report)
        self.assertIn("linear_attention_unsupported", report)
        self.assertIn("未进入板端 ACL smoke", report)


if __name__ == "__main__":
    unittest.main()
