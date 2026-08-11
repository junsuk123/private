from __future__ import annotations

import numpy as np

from app.realtime.device_plan import (
    CPU,
    GPU,
    NPU,
    DeviceInventory,
    enforce_equivalence,
    plan_devices,
    probe_devices,
    verify_decision_equivalence,
)


def _inventory(*devices: str) -> DeviceInventory:
    return DeviceInventory(tuple(devices), {d: d.lower() for d in devices})


def test_every_workload_is_placed_on_a_cpu_only_machine() -> None:
    """A PC with no accelerator must get a complete plan, not a degraded one."""
    placements = plan_devices(_inventory(CPU))

    assert placements
    assert all(item.device == CPU for item in placements)
    assert all(item.workload for item in placements)


def test_ladder_walks_past_a_missing_accelerator() -> None:
    gpu_only = {p.workload: p.device for p in plan_devices(_inventory(CPU, GPU))}
    npu_only = {p.workload: p.device for p in plan_devices(_inventory(CPU, NPU))}

    # Batch scoring prefers GPU, latency work prefers NPU; each falls to the other
    # before falling to CPU.
    assert gpu_only["ontology_candidate_scorer"] == GPU
    assert gpu_only["strategy_utility_rgcn_shadow"] == GPU
    assert npu_only["ontology_candidate_scorer"] == NPU
    assert npu_only["strategy_utility_rgcn_shadow"] == NPU


def test_decision_affecting_scorer_stays_on_its_verified_device() -> None:
    """GPU is the right shape for 4096-candidate batches, but compatibility_score
    is multiplied into weighted_utility, and GPU was measured disagreeing with CPU
    on the R-GCN. Throughput does not buy the right to move a decision kernel."""
    full = {p.workload: p.device for p in plan_devices(_inventory(CPU, GPU, NPU))}

    assert full["ontology_candidate_scorer"] == NPU


def test_decision_workloads_are_never_accelerated() -> None:
    """The property the whole module exists to protect.

    A decision that depends on which silicon was free is not reproducible, and this
    system has to be able to explain an order months later.
    """
    for inventory in (_inventory(CPU), _inventory(CPU, GPU, NPU)):
        by_key = {p.workload: p for p in plan_devices(inventory)}
        for key in (
            "trading_decision",
            "risk_checks",
            "ontology_rules",
            "strategy_utility_rgcn_decision",
        ):
            assert by_key[key].device == CPU
            assert by_key[key].accelerable is False


def test_an_override_cannot_accelerate_a_decision_workload() -> None:
    placements = {
        p.workload: p
        for p in plan_devices(
            _inventory(CPU, GPU, NPU), overrides={"trading_decision": NPU}
        )
    }

    assert placements["trading_decision"].device == CPU
    assert "not accelerable" in (placements["trading_decision"].fallback_reason or "")


def test_equivalence_compares_the_decision_not_the_bits() -> None:
    """Different silicon reorders float accumulation; demanding identical bits would
    reject every accelerator over a difference that changes nothing. What must not
    change is the decision."""
    reference = lambda s: np.asarray([s, -s])
    jittered = lambda s: np.asarray([s + 1e-9, -s - 1e-9])
    sign = lambda v: tuple((np.asarray(v) > 0).tolist())

    result = verify_decision_equivalence(
        reference, jittered, [1.0, 2.0, 3.0], decision=sign
    )

    assert result.equivalent is True
    assert result.decisions_compared == 3
    assert result.max_absolute_difference is not None


def test_a_single_flipped_decision_fails_the_check() -> None:
    """No "mostly agrees": one flip in a thousand makes the system unexplainable,
    and the cost of refusing is only that we run on CPU."""
    reference = lambda s: np.asarray([s])
    flipper = lambda s: np.asarray([-s if s == 2.0 else s])
    sign = lambda v: tuple((np.asarray(v) > 0).tolist())

    result = verify_decision_equivalence(
        reference, flipper, [1.0, 2.0, 3.0], decision=sign
    )

    assert result.equivalent is False
    assert result.decisions_disagreeing == 1


def test_a_failing_accelerator_is_demoted_to_cpu() -> None:
    placement = next(
        p
        for p in plan_devices(_inventory(CPU, GPU, NPU))
        if p.workload == "strategy_utility_rgcn_shadow"
    )
    assert placement.device != CPU

    bad = verify_decision_equivalence(
        lambda s: np.asarray([s]),
        lambda s: np.asarray([-s]),
        [1.0],
        decision=lambda v: tuple((np.asarray(v) > 0).tolist()),
    )
    demoted = enforce_equivalence(placement, bad)

    assert demoted.device == CPU
    assert "rejected" in (demoted.fallback_reason or "")


def test_a_raising_accelerator_is_not_equivalent() -> None:
    def explode(_sample):
        raise RuntimeError("device dropped")

    result = verify_decision_equivalence(lambda s: np.asarray([s]), explode, [1.0])

    assert result.equivalent is False
    assert "raised" in result.detail


def test_probe_never_raises_and_always_offers_cpu() -> None:
    inventory = probe_devices()

    assert CPU in inventory.available
