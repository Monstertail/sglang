"""Deferred prefill-stats publication: ownership, equivalence and flush ordering.

``report_prefill_stats(defer=True)`` records its publication calls so the dLLM
async loop can replay them after the next forward is launched. The recorded
arguments must be detached from the live scheduler state, and the default
(``defer=False``) callers must observe the unchanged eager sequence.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
    _CacheHitRateWindow,
)
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    QueueCount,
    SchedulerStats,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _RecordingCollector:
    """Observes the publication sequence a real collector would see."""

    def __init__(self, fail_on_log_stats=False, calls=None):
        self.calls = [] if calls is None else calls
        self.fail_on_log_stats = fail_on_log_stats

    def increment_prefill_cuda_graph_pass(self, value):
        self.calls.append(("cuda_graph_pass", value))

    def increment_realtime_tokens(
        self,
        dp_cooperation_info=None,
        prefill_compute_tokens=0,
        prefill_cache_tokens=0,
        decode_tokens=0,
    ):
        self.calls.append(
            (
                "realtime_tokens",
                prefill_compute_tokens,
                prefill_cache_tokens,
                dp_cooperation_info,
            )
        )

    def increment_estimated_perf(self, **kwargs):
        self.calls.append(("estimated_perf", kwargs))

    def increment_effective_prefill_tokens(
        self, input_tokens, device_hit_tokens, host_hit_tokens, storage_hit_tokens
    ):
        self.calls.append(("effective_prefill_tokens", input_tokens))

    def log_stats(self, stats):
        self.calls.append(("log_stats", stats))
        if self.fail_on_log_stats:
            raise RuntimeError("collector is down")


class _PoolStats:
    def get_prefill_usage_msg_parts(self):
        return ["token usage: 0.00"]

    def update_scheduler_stats(self, stats):
        stats.token_usage = 0.5


def _reporter(collector, *, priority_enabled=False):
    reporter = object.__new__(SchedulerMetricsReporter)
    reporter.is_stats_logging_rank = True
    reporter.current_scheduler_metrics_enabled = True
    reporter.enable_mfu_metrics = False
    reporter.metrics_collector = collector
    reporter.stats = SchedulerStats()
    reporter._pending_stats_calls = None
    reporter.last_prefill_stats_tic = 0.0
    reporter.last_input_throughput = 0.0
    reporter.num_retracted_reqs = 0
    reporter.num_paused_reqs = 0
    # Not NaN: the eager/deferred stats snapshots are compared by value.
    reporter.fwd_occupancy = 0.0
    reporter._graph_backend_label = "cuda graph"
    reporter.cache_hit_rate_window = _CacheHitRateWindow()
    reporter.recent_cache_hit_rate = 0.0
    reporter.kv_events = []
    reporter.scheduler = SimpleNamespace(
        forward_ct=7,
        waiting_queue=[],
        grammar_manager=[],
        disaggregation_mode=DisaggregationMode.NULL,
        enable_priority_scheduling=priority_enabled,
        enable_lora=False,
        enable_hierarchical_cache=False,
        pool_stats_observer=SimpleNamespace(get_pool_stats=_PoolStats),
        kv_events_publisher=SimpleNamespace(
            emit_kv_metrics=lambda: reporter.kv_events.append("emit_kv_metrics"),
            publish_kv_events=lambda: reporter.kv_events.append("publish_kv_events"),
        ),
    )
    return reporter


def _enable_mfu(reporter):
    """Unit perf constants: the estimator's arithmetic is not under test here."""
    reporter.enable_mfu_metrics = True
    reporter._linear_flops_per_token = 1.0
    reporter._attn_dot_flops_coeff = 1.0
    reporter._kv_cache_bytes_per_token = 1.0
    reporter._weight_read_bytes_per_token = 1.0
    reporter._qkv_act_bytes_per_token = 1.0
    reporter._ffn_act_bytes_per_token = 1.0
    reporter._prefill_attn_act_read_per_token = 1.0


def _prefill_stats(priority_enabled=False, input_tokens=64):
    return PrefillStats(
        log_input_tokens=input_tokens,
        log_hit_tokens=16,
        new_token_ratio=0.5,
        num_running_reqs=QueueCount(
            total=2, by_priority={} if priority_enabled else None
        ),
        num_new_seqs=2,
    )


def _patched_disagg():
    return patch(
        "sglang.srt.managers.scheduler_components.metrics_reporter.get_disagg",
        return_value=SimpleNamespace(
            language_only=False, encoder_transfer_backend="mooncake"
        ),
    )


def _report(reporter, *, defer, dp_cooperation_info=None, batch=None, input_tokens=64):
    with _patched_disagg():
        reporter.report_prefill_stats(
            batch=batch if batch is not None else SimpleNamespace(forward_iter=3),
            prefill_stats=_prefill_stats(
                reporter.scheduler.enable_priority_scheduling, input_tokens
            ),
            can_run_cuda_graph=True,
            dp_cooperation_info=dp_cooperation_info,
            defer=defer,
        )


class TestDeferredPrefillStats(unittest.TestCase):
    def test_default_callers_publish_eagerly_by_identity(self):
        """AR / PD-prefill callers keep the unchanged call sequence and objects."""
        collector = _RecordingCollector()
        reporter = _reporter(collector)
        dp_info = DPCooperationInfo(num_prefill_ranks=1)

        _report(reporter, defer=False, dp_cooperation_info=dp_info)

        self.assertEqual(
            [call[0] for call in collector.calls],
            [
                "cuda_graph_pass",
                "realtime_tokens",
                "effective_prefill_tokens",
                "log_stats",
            ],
        )
        self.assertIs(collector.calls[1][3], dp_info)
        self.assertIs(collector.calls[3][1], reporter.stats)
        self.assertIsNone(reporter._pending_stats_calls)
        # A flush on a non-deferring reporter must be a no-op.
        reporter.flush_deferred_stats()
        self.assertEqual(len(collector.calls), 4)

    def test_deferred_replay_matches_the_eager_publication(self):
        eager_collector = _RecordingCollector()
        _report(_reporter(eager_collector), defer=False)

        deferred_collector = _RecordingCollector()
        reporter = _reporter(deferred_collector)
        _report(reporter, defer=True)

        # Nothing published yet, but KV metrics and events stayed eager.
        self.assertEqual(deferred_collector.calls, [])
        self.assertEqual(reporter.kv_events, ["emit_kv_metrics", "publish_kv_events"])

        reporter.flush_deferred_stats()
        self.assertEqual(
            [call[:3] for call in deferred_collector.calls],
            [call[:3] for call in eager_collector.calls],
        )
        self.assertIsNone(reporter._pending_stats_calls)

    def test_the_log_line_is_published_before_the_collector_calls(self):
        """The server log keeps its position both eagerly and on replay."""
        for defer in (False, True):
            with self.subTest(defer=defer):
                observed = []
                reporter = _reporter(_RecordingCollector(calls=observed))
                stub_logger = SimpleNamespace(
                    info=lambda msg: observed.append(("log", msg))
                )
                with patch(
                    "sglang.srt.managers.scheduler_components.metrics_reporter.logger",
                    stub_logger,
                ):
                    _report(reporter, defer=defer)
                    reporter.flush_deferred_stats()

                self.assertEqual(observed[0][0], "log")
                self.assertTrue(observed[0][1].startswith("Prefill batch"))
                self.assertEqual(observed[-1][0], "log_stats")

    def test_deferred_mfu_metrics_keep_their_position_and_own_their_values(self):
        collector = _RecordingCollector()
        reporter = _reporter(collector)
        _enable_mfu(reporter)
        batch = SimpleNamespace(forward_iter=3, extend_lens=[4, 8], prefix_lens=[0, 16])

        _report(reporter, defer=True, batch=batch)
        batch.extend_lens = [1024, 1024]
        reporter.flush_deferred_stats()

        self.assertEqual(
            [call[0] for call in collector.calls],
            [
                "cuda_graph_pass",
                "realtime_tokens",
                "estimated_perf",
                "effective_prefill_tokens",
                "log_stats",
            ],
        )
        perf_kwargs = collector.calls[2][1]
        self.assertEqual(
            sorted(perf_kwargs),
            ["num_flops_per_gpu", "num_read_bytes_per_gpu", "num_write_bytes_per_gpu"],
        )
        for value in perf_kwargs.values():
            self.assertIsInstance(value, float)
        # Estimated from the 12 tokens the batch carried at report time.
        self.assertEqual(perf_kwargs["num_write_bytes_per_gpu"], 36.0)

    def test_a_rank_that_reports_nothing_records_nothing(self):
        """The early return must not leave a record the next flush would replay."""
        collector = _RecordingCollector()
        reporter = _reporter(collector)
        reporter.is_stats_logging_rank = False
        reporter.current_scheduler_metrics_enabled = False

        _report(reporter, defer=True)

        self.assertEqual(collector.calls, [])
        self.assertIsNone(reporter._pending_stats_calls)
        reporter.flush_deferred_stats()
        self.assertEqual(collector.calls, [])

    def test_deferred_record_owns_its_arguments(self):
        """Later scheduler mutation must not reach the replayed values."""
        collector = _RecordingCollector()
        reporter = _reporter(collector)
        dp_info = DPCooperationInfo(num_prefill_ranks=1)
        batch = SimpleNamespace(forward_iter=3)

        _report(reporter, defer=True, dp_cooperation_info=dp_info, batch=batch)

        recorded = [
            value
            for _, args, kwargs in reporter._pending_stats_calls
            for value in (*args, *kwargs.values())
        ]
        self.assertNotIn(id(batch), [id(value) for value in recorded])

        reporter.stats.token_usage = 0.99
        reporter.stats.num_running_reqs.total = 999
        reporter.stats.routing_key_running_req_counts.append(5)
        dp_info.num_prefill_ranks = 42

        reporter.flush_deferred_stats()
        published_stats = collector.calls[-1][1]
        self.assertEqual(published_stats.token_usage, 0.5)
        self.assertEqual(published_stats.num_running_reqs.total, 2)
        self.assertEqual(published_stats.routing_key_running_req_counts, [])
        self.assertEqual(collector.calls[1][3].num_prefill_ranks, 1)

    def test_absent_dp_cooperation_info_is_deferred_unchanged(self):
        """The dLLM TP1/DP1 path always passes None; cloning must not run on it."""
        collector = _RecordingCollector()
        reporter = _reporter(collector)

        _report(reporter, defer=True, dp_cooperation_info=None)
        reporter.flush_deferred_stats()

        self.assertIsNone(collector.calls[1][3])

    def test_deferred_priority_breakdown_keeps_empty_and_absent_maps_distinct(self):
        """{} zeroes the known-priority gauges; None skips the breakdown."""
        collector = _RecordingCollector()
        reporter = _reporter(collector, priority_enabled=True)
        reporter.stats.num_queue_reqs = QueueCount(total=0, by_priority={})
        reporter.stats.num_decode_prealloc_queue_reqs = QueueCount(
            total=0, by_priority=None
        )
        live_map = {3: 1}
        reporter.stats.num_prefill_bootstrap_queue_reqs = QueueCount(
            total=1, by_priority=live_map
        )

        _report(reporter, defer=True)
        live_map[3] = 77
        reporter.flush_deferred_stats()

        published_stats = collector.calls[-1][1]
        self.assertEqual(published_stats.num_queue_reqs.by_priority, {})
        self.assertIsNone(published_stats.num_decode_prealloc_queue_reqs.by_priority)
        self.assertEqual(
            published_stats.num_prefill_bootstrap_queue_reqs.by_priority, {3: 1}
        )

    def test_a_failing_replay_does_not_republish_or_retain_the_record(self):
        collector = _RecordingCollector(fail_on_log_stats=True)
        reporter = _reporter(collector)

        _report(reporter, defer=True)
        with self.assertRaises(RuntimeError):
            reporter.flush_deferred_stats()

        self.assertIsNone(reporter._pending_stats_calls)
        published = list(collector.calls)
        reporter.flush_deferred_stats()
        self.assertEqual(collector.calls, published)

    def test_consecutive_rounds_publish_each_record_exactly_once(self):
        """The loop order is record(k) -> launch(k+1) -> flush(k) -> report(k+1)."""
        collector = _RecordingCollector()
        reporter = _reporter(collector)

        self.assertIsNone(reporter._pending_stats_calls)
        _report(reporter, defer=True, input_tokens=11)
        self.assertEqual(collector.calls, [])

        reporter.flush_deferred_stats()
        self.assertIsNone(reporter._pending_stats_calls)
        after_round_k = list(collector.calls)
        self.assertEqual(
            [call[0] for call in after_round_k],
            [
                "cuda_graph_pass",
                "realtime_tokens",
                "effective_prefill_tokens",
                "log_stats",
            ],
        )

        # The next report finds the slot empty and records round k+1 on its own.
        _report(reporter, defer=True, input_tokens=22)
        self.assertEqual(collector.calls, after_round_k)
        self.assertIsNotNone(reporter._pending_stats_calls)

        reporter.flush_deferred_stats()
        self.assertEqual(
            [call[1] for call in collector.calls if call[0] == "realtime_tokens"],
            [11, 22],
        )
        self.assertEqual(len(collector.calls), 2 * len(after_round_k))

    def test_second_deferred_report_is_rejected_before_overwriting_the_first(self):
        collector = _RecordingCollector()
        reporter = _reporter(collector)

        _report(reporter, defer=True, input_tokens=11)
        pending = reporter._pending_stats_calls

        with self.assertRaisesRegex(RuntimeError, "was not flushed"):
            _report(reporter, defer=True, input_tokens=22)

        self.assertIs(reporter._pending_stats_calls, pending)
        reporter.flush_deferred_stats()
        self.assertEqual(
            [call[1] for call in collector.calls if call[0] == "realtime_tokens"],
            [11],
        )


class _StubDllmScheduler(SchedulerDllmMixin):
    """Drives the real dLLM event loop through its four flush branches.

    Round 1 delivers a control request that pauses the engine; round 2 finds no
    batch to run and exits, so the control-request, pause, idle and loop-exit
    flush sites each run exactly once with a record pending.
    """

    def __init__(self, reporter, events):
        self.metrics_reporter = reporter
        self.events = events
        self.dllm_config = SimpleNamespace(
            algorithm="JointThreshold",
            first_done_first_out_mode=True,
            algorithm_config={"vectorized_decoding": True},
        )
        self.ps = SimpleNamespace(tp_size=1, pp_size=1, dp_size=1)
        self.enable_overlap = False
        self.enable_pdmux = False
        self.disaggregation_mode = DisaggregationMode.NULL
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.rust_server = None
        self.enable_lora = False
        self.output_streamer = SimpleNamespace(has_additional_customized_info=False)
        self.request_receiver = SimpleNamespace(recv_requests=self._recv_requests)
        self.invariant_checker = SimpleNamespace(self_check_during_busy=lambda: None)
        self.gracefully_exit = False
        self._engine_paused = False
        self._sched_idled = False
        self.running_batch = None
        self.last_batch = None
        self.cur_batch_for_debug = None
        self._round = 0
        self.armed = 0

    def arm(self):
        """Record one round of stats; its token count identifies the record."""
        self.armed += 1
        _report(self.metrics_reporter, defer=True, input_tokens=self.armed)

    def _recv_requests(self):
        self._round += 1
        return ["control-request"] if self._round == 1 else []

    def process_input_requests(self, recv_reqs):
        self._engine_paused = bool(recv_reqs)
        if recv_reqs:
            self.arm()

    def get_next_batch_to_run(self, running_batch, last_batch):
        self.arm()
        return SimpleNamespace(running_batch=running_batch, batch_to_run=None)

    def on_idle(self):
        self.arm()
        self.gracefully_exit = True


class TestDllmFlushOrdering(unittest.TestCase):
    def test_materialize_path_flushes_after_copy_enqueue_before_wait(self):
        events = []

        class _CopyDone:
            def record(self):
                events.append("record_copy_done")

            def synchronize(self):
                events.append("wait_copy")

        class _DeferredResult:
            def map_device_tensors(self, copy_tensor):
                events.append("enqueue_d2h")

            def materialize(self, copy_done):
                events.append("materialize")
                return [[1, 2]], [0], [{"round": 1}]

        copy_done = _CopyDone()
        scheduler = SimpleNamespace(
            copy_stream=SimpleNamespace(
                wait_stream=lambda stream: events.append("order_copy_stream")
            ),
            copy_stream_ctx=nullcontext(),
            device_module=SimpleNamespace(
                current_stream=lambda: object(), Event=lambda: copy_done
            ),
            metrics_reporter=SimpleNamespace(
                flush_deferred_stats=lambda: events.append("stats")
            ),
        )
        scheduler._flush_dllm_stats = lambda: SchedulerDllmMixin._flush_dllm_stats(
            scheduler
        )
        result = SimpleNamespace(
            dllm_deferred_result=_DeferredResult(),
            copy_done=None,
            next_token_ids=None,
            accept_length_per_req_cpu=None,
            dllm_algo_state=None,
        )

        SchedulerDllmMixin._materialize_dllm_result(scheduler, result)

        self.assertEqual(
            events,
            [
                "order_copy_stream",
                "enqueue_d2h",
                "record_copy_done",
                "stats",
                "wait_copy",
                "materialize",
            ],
        )
        self.assertEqual(result.next_token_ids, [[1, 2]])
        self.assertEqual(result.accept_length_per_req_cpu, [0])
        self.assertEqual(result.dllm_algo_state, [{"round": 1}])
        self.assertIsNone(result.dllm_deferred_result)

    def test_every_flush_site_publishes_the_pending_record(self):
        events = []
        reporter = _reporter(_RecordingCollector(calls=events))
        stub = _StubDllmScheduler(reporter, events)
        stub.arm()

        with patch("sglang.srt.dllm.mixin.scheduler.is_cuda", return_value=True):
            stub.event_loop_dllm_async()

        # One record per flush site: control request, pause, idle and loop exit.
        # The token count identifies the round each record came from.
        self.assertEqual(
            [
                event
                for event in events
                if isinstance(event, tuple) and event[0] == "realtime_tokens"
            ],
            [
                ("realtime_tokens", 1, 16, None),
                ("realtime_tokens", 2, 16, None),
                ("realtime_tokens", 3, 16, None),
                ("realtime_tokens", 4, 16, None),
            ],
        )
        self.assertIsNone(reporter._pending_stats_calls)


if __name__ == "__main__":
    unittest.main()
