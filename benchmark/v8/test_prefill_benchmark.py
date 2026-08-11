import io
from contextlib import redirect_stdout
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

from benchmark.v8.prefill_benchmark import (
    Case,
    TimeResult,
    build_cases,
    critical_path_median,
    derive_rank_topology,
    divisors,
    execution_heads_for_local,
    has_complete_torchrun_env,
    launcher_watchdog_seconds,
    make_parallel_case,
    maybe_launch_distributed,
    parse_args,
    print_result_matrix,
    rerange_cp_rank_major,
    result_row,
    selected_backends,
    write_csv,
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

    def test_two_gpu_default_uses_physical_tp_and_dsa_cp_endpoints(self):
        rank0_args = parse_args([], distributed_world_size=2, distributed_rank=0)
        rank1_args = parse_args([], distributed_world_size=2, distributed_rank=1)
        rank0_cases = build_cases(rank0_args)
        rank1_cases = build_cases(rank1_args)

        self.assertEqual(len(rank0_cases), 12)
        self.assertEqual(rank0_args.tp, [2])
        self.assertEqual(rank0_args.cp, [1, 2])
        self.assertEqual(
            [(case.cp_size, case.heads) for case in rank0_cases[:2]],
            [(1, 64), (2, 128)],
        )
        self.assertEqual(
            [(case.cp_rank, case.attention_tp_rank) for case in rank0_cases[:2]],
            [(0, 0), (0, 0)],
        )
        self.assertEqual(
            [(case.cp_rank, case.attention_tp_rank) for case in rank1_cases[:2]],
            [(0, 1), (1, 0)],
        )
        self.assertTrue(all(case.physical for case in rank0_cases + rank1_cases))

    def test_physical_round_robin_uses_runtime_padding(self):
        expected = {
            1: (880, 875, 5, 16, 64),
            2: (438, 437, 1, 32, 64),
            4: (220, 218, 2, 64, 64),
            8: (110, 109, 1, 128, 128),
        }
        args = parse_args(
            ["--cp", "--batch", "1"],
            distributed_world_size=8,
            distributed_rank=7,
        )
        cases = build_cases(args)

        self.assertEqual(args.cp, [1, 2, 4, 8])
        for case in cases:
            rows, logical, padding, local_heads, execution_heads = expected[
                case.cp_size
            ]
            self.assertEqual(case.rows, rows)
            self.assertEqual(case.logical_rows, logical)
            self.assertEqual(case.padding_rows, padding)
            self.assertEqual(case.logical_heads, local_heads)
            self.assertEqual(case.heads, execution_heads)

    def test_physical_cp_partitions_the_flattened_batch_not_each_request(self):
        rows = []
        for rank in range(8):
            args = parse_args(
                ["--tp", "8", "--cp", "8", "--batch", "8", "--new", "1"],
                distributed_world_size=8,
                distributed_rank=rank,
            )
            case = build_cases(args)[0]
            rows.append((case.logical_rows, case.rows, case.padding_rows))
        self.assertEqual(rows, [(1, 2, 1)] * 8)

    def test_rank_topology_matches_sglang_group_layout(self):
        topology = derive_rank_topology(
            world_size=8,
            total_tp=8,
            cp_size=2,
            global_heads=128,
            global_rank=5,
        )
        self.assertEqual(topology.attention_tp_size, 4)
        self.assertEqual(topology.cp_rank, 1)
        self.assertEqual(topology.attention_tp_rank, 1)
        self.assertEqual(topology.attention_tp_ranks, (4, 5, 6, 7))
        self.assertEqual(topology.cp_ranks, (1, 5))
        self.assertEqual((topology.head_start, topology.head_stop), (32, 64))

        coordinates = {
            (
                derive_rank_topology(8, 8, 2, 128, rank).cp_rank,
                derive_rank_topology(8, 8, 2, 128, rank).attention_tp_rank,
            )
            for rank in range(8)
        }
        self.assertEqual(coordinates, {(cp, tp) for cp in range(2) for tp in range(4)})

    def test_physical_job_rejects_manual_rank_and_world_mismatch(self):
        with self.assertRaisesRegex(ValueError, "derives CP rank"):
            parse_args(
                ["--cp-rank", "0"],
                distributed_world_size=2,
                distributed_rank=0,
            )
        with self.assertRaisesRegex(ValueError, "equal WORLD_SIZE"):
            parse_args(
                ["--tp", "1"],
                distributed_world_size=2,
                distributed_rank=0,
            )

    def test_critical_path_reducer_is_iteration_max_then_median(self):
        samples = [
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0],
        ]
        self.assertEqual(critical_path_median(samples), 10.0)
        self.assertEqual(max(sorted(rank)[1] for rank in samples), 1.0)
        with self.assertRaises(ValueError):
            critical_path_median([[1.0, float("nan")]])

    def test_cp_rerange_matches_sglang_transpose_reshape_order(self):
        rank_major = torch.tensor([[10], [0], [20], [0]], dtype=torch.int32)
        token_major = rerange_cp_rank_major(rank_major, cp_size=2)
        self.assertEqual(token_major.flatten().tolist(), [10, 20, 0, 0])
        with self.assertRaises(ValueError):
            rerange_cp_rank_major(rank_major, cp_size=3)

    def test_divisors_drive_physical_cp_sweep(self):
        self.assertEqual(divisors(8), [1, 2, 4, 8])

    def test_partial_torchrun_environment_is_rejected(self):
        with mock.patch.dict(os.environ, {"RANK": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "partial torchrun environment"):
                has_complete_torchrun_env()
        complete = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with mock.patch.dict(os.environ, complete, clear=True):
            self.assertTrue(has_complete_torchrun_env())

    def test_auto_launcher_propagates_worker_exit_and_safety_environment(self):
        process = mock.Mock()
        process.wait.return_value = 7
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("torch.cuda.device_count", return_value=2),
            mock.patch(
                "benchmark.v8.prefill_benchmark.subprocess.Popen",
                return_value=process,
            ) as popen,
        ):
            self.assertEqual(maybe_launch_distributed(["--batch", "1"]), 7)
        command = popen.call_args.args[0]
        environment = popen.call_args.kwargs["env"]
        self.assertIn("--nproc-per-node=2", command)
        self.assertEqual(command.count("--distributed-worker"), 1)
        self.assertEqual(environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "1")
        self.assertEqual(environment["TORCH_NCCL_BLOCKING_WAIT"], "1")
        process.wait.assert_called_once_with(timeout=2 * 60 * 60)

    def test_auto_launcher_watchdog_terminates_the_worker_process_group(self):
        process = mock.Mock(pid=4321)
        process.wait.side_effect = [
            subprocess.TimeoutExpired("torchrun", 5),
            0,
        ]
        process.poll.return_value = None
        with (
            mock.patch.dict(
                os.environ,
                {"MLA_PREFILL_BENCH_WATCHDOG_SECONDS": "5"},
                clear=True,
            ),
            mock.patch("torch.cuda.device_count", return_value=2),
            mock.patch(
                "benchmark.v8.prefill_benchmark.subprocess.Popen",
                return_value=process,
            ),
            mock.patch("benchmark.v8.prefill_benchmark.os.killpg") as killpg,
            redirect_stdout(io.StringIO()),
        ):
            with mock.patch("sys.stderr", io.StringIO()):
                self.assertEqual(maybe_launch_distributed([]), 124)
        killpg.assert_called_once_with(4321, signal.SIGTERM)

    def test_watchdog_and_timing_inputs_are_validated(self):
        with mock.patch.dict(
            os.environ,
            {"MLA_PREFILL_BENCH_WATCHDOG_SECONDS": "bad"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                launcher_watchdog_seconds()
        for argv, message in (
            (["--iters", "0"], "--iters must be positive"),
            (["--warmup", "-1"], "--warmup must be non-negative"),
            (["--flush-mb", "-1"], "--flush-mb must be non-negative"),
            (["--limit", "-1"], "--limit must be non-negative"),
        ):
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, message):
                    parse_args(argv)

    def test_csv_write_is_atomic_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            with redirect_stdout(io.StringIO()):
                write_csv(str(output), [{"case": "h64", "ms": "1.0"}])
            self.assertEqual(output.read_text().splitlines(), ["case,ms", "h64,1.0"])
            self.assertEqual(list(output.parent.glob(f".{output.name}.tmp.*")), [])

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
        self.assertEqual(row["kernel_ms"], "1.250000")
        self.assertEqual(row["cp_allgather_probe_plus_kernel_ms"], "")

        output = io.StringIO()
        with redirect_stdout(output):
            print_result_matrix([row], {"v8"})
        self.assertIn("1.2500", output.getvalue())
        self.assertIn("Hglobal", output.getvalue())

    def test_physical_csv_records_world_row_ranges(self):
        case = make_parallel_case(
            "physical-schema",
            "v32",
            batch=1,
            new_tokens=875,
            cached_len=69125,
            tp_size=8,
            cp_size=8,
            global_heads=128,
            topk=2048,
            cp_rank=7,
            physical=True,
            global_rank=7,
            world_size=8,
            attention_tp_rank=0,
        )
        row = result_row(case, TimeResult("v8", True, 1.0, kernel_ms=1.0))
        self.assertEqual(row["logical_rows"], "109")
        self.assertEqual(row["logical_rows_min"], "109")
        self.assertEqual(row["logical_rows_max"], "110")
        self.assertEqual(row["padding_rows_min"], "0")
        self.assertEqual(row["padding_rows_max"], "1")
        self.assertEqual(row["rank_scope"], "all-ranks")
        self.assertEqual(row["cp_allgather_probe_plus_kernel_ms"], "")

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
