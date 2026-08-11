import io
from contextlib import redirect_stdout
import unittest

from benchmark.v8.prefill_benchmark import (
    Case,
    TimeResult,
    build_cases,
    execution_heads_for_local,
    make_parallel_case,
    parse_args,
    print_result_matrix,
    result_row,
    selected_backends,
)


class ParallelCaseTest(unittest.TestCase):
    def test_no_arguments_runs_h64_and_h128_for_six_batches(self):
        args = parse_args([])
        cases = build_cases(args)

        self.assertEqual(len(cases), 12)
        self.assertEqual(args.batch, [1, 2, 4, 8, 16, 32])
        self.assertEqual(args.tp, [2, 1])
        self.assertEqual(args.cp, [1])
        self.assertEqual(
            [(case.batch, case.heads) for case in cases],
            [
                (batch, heads)
                for batch in [1, 2, 4, 8, 16, 32]
                for heads in [64, 128]
            ],
        )
        self.assertEqual(selected_backends(args), ("decode", "v8"))

    def test_bare_tp_and_cp_select_documented_sweeps(self):
        tp_args = parse_args(["--tp"])
        self.assertEqual(tp_args.tp, [1, 2, 4, 8])
        self.assertEqual(tp_args.cp, [1])
        self.assertEqual(len(build_cases(tp_args)), 24)

        cp_args = parse_args(["--cp"])
        self.assertEqual(cp_args.tp, [8])
        self.assertEqual(cp_args.cp, [1, 2, 4, 8])
        self.assertEqual(len(build_cases(cp_args)), 24)

        matrix_args = parse_args(["--tp", "--cp"])
        self.assertEqual(matrix_args.tp, [1, 2, 4, 8])
        self.assertEqual(matrix_args.cp, [1, 2, 4, 8])
        self.assertEqual(len(build_cases(matrix_args)), 60)

    def test_comparison_flags_add_to_decode_and_v8(self):
        args = parse_args(
            ["--bf16", "--fp8-dequant", "--trtllm", "--decode-chunked"]
        )
        self.assertEqual(
            selected_backends(args),
            (
                "decode",
                "v8",
                "bf16",
                "fp8_dequant",
                "trtllm",
                "decode_chunked",
            ),
        )

    def test_fixed_total_tp_cp_sweep_reaches_h128(self):
        args = parse_args(
            [
                "--suite",
                "custom",
                "--layouts",
                "v32",
                "--batch",
                "1",
                "--new",
                "10000",
                "--cached",
                "90000",
                "--tp",
                "8",
                "--cp",
                "1",
                "2",
                "4",
                "8",
                "--global-heads",
                "128",
                "--topk",
                "2048",
            ]
        )
        cases = build_cases(args)

        self.assertEqual([case.tp_size for case in cases], [8, 8, 8, 8])
        self.assertEqual(
            [case.attention_tp_size for case in cases], [8, 4, 2, 1]
        )
        self.assertEqual(
            [case.logical_heads for case in cases], [16, 32, 64, 128]
        )
        self.assertEqual([case.heads for case in cases], [64, 64, 64, 128])
        self.assertEqual(
            [case.q_per_rank for case in cases], [10000, 5000, 2500, 1250]
        )

    def test_legacy_local_heads_infers_total_tp_per_cp(self):
        args = parse_args(
            [
                "--suite",
                "custom",
                "--batch",
                "1",
                "--new",
                "10000",
                "--cached",
                "90000",
                "--cp",
                "1",
                "8",
                "--heads",
                "64",
                "--global-heads",
                "128",
                "--topk",
                "2048",
            ]
        )
        cases = build_cases(args)

        self.assertEqual([case.tp_size for case in cases], [2, 16])
        self.assertEqual([case.attention_tp_size for case in cases], [2, 2])
        self.assertEqual([case.logical_heads for case in cases], [64, 64])
        self.assertEqual([case.heads for case in cases], [64, 64])

    def test_non_divisible_cp_uses_quotient_remainder_distribution(self):
        cases = [
            make_parallel_case(
                "uneven",
                "v32",
                batch=1,
                new_tokens=875,
                cached_len=69125,
                tp_size=8,
                cp_size=8,
                global_heads=128,
                topk=2048,
                cp_rank=rank,
            )
            for rank in range(8)
        ]

        self.assertEqual(
            [case.q_per_rank for case in cases],
            [110, 110, 110, 109, 109, 109, 109, 109],
        )
        self.assertEqual(sum(case.q_per_rank for case in cases), 875)
        self.assertTrue(all(case.cp_remainder == 3 for case in cases))

    def test_total_tp_must_include_cp(self):
        with self.assertRaisesRegex(ValueError, "total TP 2 is not divisible"):
            make_parallel_case(
                "invalid",
                "v32",
                batch=1,
                new_tokens=10000,
                cached_len=90000,
                tp_size=2,
                cp_size=4,
                global_heads=128,
                topk=2048,
            )

    def test_multi_tp_cp_matrix_skips_only_invalid_pairs(self):
        args = parse_args(
            [
                "--suite",
                "custom",
                "--batch",
                "1",
                "--new",
                "10000",
                "--cached",
                "90000",
                "--global-heads",
                "128",
                "--tp",
                "1",
                "2",
                "4",
                "8",
                "--cp",
                "1",
                "2",
                "4",
                "8",
                "--cp-rank",
                "0",
                "--topk",
                "2048",
            ]
        )
        cases = build_cases(args)

        self.assertEqual(
            [(case.tp_size, case.cp_size) for case in cases],
            [
                (1, 1),
                (2, 1),
                (4, 1),
                (8, 1),
                (2, 2),
                (4, 2),
                (8, 2),
                (4, 4),
                (8, 4),
                (8, 8),
            ],
        )
        self.assertEqual(
            [(skip.tp_size, skip.cp_size) for skip in args.parallel_skips],
            [(1, 2), (1, 4), (2, 4), (1, 8), (2, 8), (4, 8)],
        )
        self.assertEqual(
            args.valid_parallel_configs,
            (
                (128, 1, 1),
                (128, 2, 1),
                (128, 2, 2),
                (128, 4, 1),
                (128, 4, 2),
                (128, 4, 4),
                (128, 8, 1),
                (128, 8, 2),
                (128, 8, 4),
                (128, 8, 8),
            ),
        )
        self.assertTrue(
            all("not divisible" in skip.reason for skip in args.parallel_skips)
        )

    def test_zero_valid_parallel_matrix_is_an_error_with_metadata(self):
        args = parse_args(
            [
                "--suite",
                "custom",
                "--batch",
                "1",
                "--new",
                "10000",
                "--cached",
                "90000",
                "--global-heads",
                "128",
                "--tp",
                "1",
                "--cp",
                "2",
                "--cp-rank",
                "0",
                "--topk",
                "2048",
            ]
        )
        with self.assertRaisesRegex(ValueError, "no runnable benchmark cases"):
            build_cases(args)
        self.assertEqual(len(args.parallel_skips), 1)
        self.assertEqual(args.parallel_skips[0].tp_size, 1)
        self.assertEqual(args.parallel_skips[0].cp_size, 2)

    def test_cp_rank_matrix_skips_out_of_range_ranks(self):
        args = parse_args(
            [
                "--suite",
                "custom",
                "--batch",
                "1",
                "--new",
                "10000",
                "--cached",
                "90000",
                "--global-heads",
                "128",
                "--tp",
                "8",
                "--cp",
                "1",
                "2",
                "4",
                "8",
                "--cp-rank",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "--topk",
                "2048",
            ]
        )
        cases = build_cases(args)
        rank_skips = [
            skip for skip in args.skipped_configs if skip.kind == "cp_rank"
        ]

        self.assertEqual(len(cases), 1 + 2 + 4 + 8)
        self.assertEqual(len(rank_skips), 7 + 6 + 4)
        self.assertTrue(
            all(skip.cp_rank is not None for skip in rank_skips)
        )

    def test_zero_query_rank_is_skipped_and_single_rank_is_strict(self):
        matrix_args = parse_args(
            [
                "--suite",
                "custom",
                "--batch",
                "1",
                "--new",
                "1",
                "--cached",
                "8192",
                "--global-heads",
                "128",
                "--tp",
                "8",
                "--cp",
                "8",
                "--cp-rank",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "--topk",
                "2048",
            ]
        )
        cases = build_cases(matrix_args)
        empty_skips = [
            skip
            for skip in matrix_args.skipped_configs
            if skip.kind == "empty_rank"
        ]
        self.assertEqual([(case.cp_rank, case.q_per_rank) for case in cases], [(0, 1)])
        self.assertEqual(len(empty_skips), 7)

        strict_args = parse_args(
            [
                "--suite",
                "custom",
                "--batch",
                "1",
                "--new",
                "1",
                "--cached",
                "8192",
                "--global-heads",
                "128",
                "--tp",
                "8",
                "--cp",
                "8",
                "--cp-rank",
                "7",
                "--topk",
                "2048",
            ]
        )
        with self.assertRaisesRegex(ValueError, "no runnable benchmark cases"):
            build_cases(strict_args)
        self.assertEqual(strict_args.skipped_configs[0].kind, "empty_rank")

    def test_csv_schema_separates_global_local_and_execution_heads(self):
        case = make_parallel_case(
            "schema",
            "v32",
            batch=1,
            new_tokens=10000,
            cached_len=90000,
            tp_size=8,
            cp_size=2,
            global_heads=128,
            topk=2048,
        )
        row = result_row(case, TimeResult("v8", True, 1.25), 64)

        self.assertEqual(row["tp_size"], "8")
        self.assertEqual(row["attention_tp_size"], "4")
        self.assertEqual(row["cp_size"], "2")
        self.assertEqual(row["global_heads"], "128")
        self.assertEqual(row["local_heads"], "32")
        self.assertEqual(row["heads"], "32")
        self.assertEqual(row["execution_heads"], "64")
        self.assertEqual(row["heads_actual"], "64")

        output = io.StringIO()
        with redirect_stdout(output):
            print_result_matrix([row], {"v8"})
        self.assertIn("1.2500", output.getvalue())
        self.assertIn("Hglobal", output.getvalue())

    def test_execution_bucket_boundaries(self):
        self.assertEqual(execution_heads_for_local(1), 64)
        self.assertEqual(execution_heads_for_local(64), 64)
        self.assertEqual(execution_heads_for_local(65), 128)
        self.assertEqual(execution_heads_for_local(128), 128)
        with self.assertRaises(ValueError):
            execution_heads_for_local(129)

    def test_legacy_case_constructor_remains_usable(self):
        case = Case("legacy", "v32", 1, 875, 69125, 1, 64, 2048)
        self.assertEqual(case.logical_heads, 64)
        self.assertEqual(case.heads, 64)
        self.assertEqual(case.attention_tp_size, 2)


if __name__ == "__main__":
    unittest.main()
