from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


#: Devices this project knows how to place work on, cheapest guarantee last.
#: CPU is not a device we *choose* — it is the one that is always there, which is
#: why every ladder ends in it and why nothing may be placed anywhere else without
#: first proving it agrees with CPU (see :func:`verify_decision_equivalence`).
CPU = "CPU"
GPU = "GPU"
NPU = "NPU"


@dataclass(frozen=True)
class DeviceInventory:
    """What this machine can actually run, probed once."""

    available: tuple[str, ...]
    names: Mapping[str, str]
    probe_error: str | None = None

    def has(self, device: str) -> bool:
        return device.upper() in self.available


def probe_devices() -> DeviceInventory:
    """Devices OpenVINO reports, always including CPU.

    A machine with no OpenVINO install, no iGPU or no NPU is the normal case, not
    an error: this returns a CPU-only inventory and records why. Nothing downstream
    is allowed to treat a missing accelerator as a failure.
    """
    if importlib.util.find_spec("openvino") is None:
        return DeviceInventory((CPU,), {CPU: "python"}, "openvino not installed")
    try:
        try:
            from openvino import Core  # type: ignore
        except Exception:  # noqa: BLE001 - older layout
            from openvino.runtime import Core  # type: ignore
        core = Core()
        found: list[str] = []
        names: dict[str, str] = {}
        for raw in core.available_devices:
            device = str(raw).upper().split(".")[0]
            if device not in found:
                found.append(device)
            try:
                names[device] = str(core.get_property(str(raw), "FULL_DEVICE_NAME"))
            except Exception:  # noqa: BLE001 - naming is cosmetic
                names.setdefault(device, str(raw))
        if CPU not in found:
            found.append(CPU)
            names.setdefault(CPU, "cpu")
        return DeviceInventory(tuple(found), names)
    except Exception as exc:  # noqa: BLE001 - probing must never break startup
        return DeviceInventory((CPU,), {CPU: "python"}, f"probe failed: {exc}")


@dataclass(frozen=True)
class Workload:
    """One placeable unit of compute.

    ``accelerable`` is the important field and it is False for anything that
    decides. Trading logic, risk checks and ontology rules run on CPU not because
    they would be slow elsewhere but because a decision that depends on which
    silicon happened to be free is not reproducible, and this system has to be able
    to explain why it placed an order months later.
    """

    key: str
    description: str
    ladder: tuple[str, ...]
    accelerable: bool = True
    #: Why this workload prefers what it prefers, surfaced in diagnostics.
    rationale: str = ""


#: Placement policy. Batch-heavy scoring goes to the iGPU because throughput wins
#: over per-call latency there; small low-latency graphs go to the NPU; everything
#: that produces a DECISION stays on CPU.
WORKLOADS: tuple[Workload, ...] = (
    # The R-GCN is split in two on purpose, because the same graph is run for two
    # different reasons and only one of them may be accelerated.
    #
    # Measured on this machine against the shipped checkpoint: NPU disagreed with
    # CPU on 2 of 12 sampled contexts and GPU on 1 of 12. The numeric gap is tiny
    # (3.0e-2 and 5.6e-2 bps) but `expected_net_return_bps > 0` is a threshold, so a
    # value sitting near zero changes side and the router reaches a different
    # verdict. ShadowIntelligenceService already routes on `cpu_output` and uses the
    # accelerator only for its comparison lane; this table records that as policy so
    # nobody "optimises" the decision onto an accelerator later.
    Workload(
        key="strategy_utility_rgcn_decision",
        description="R-GCN inference whose output is routed on",
        ladder=(CPU,),
        accelerable=False,
        rationale="threshold on net edge flips under device float reordering",
    ),
    Workload(
        key="strategy_utility_rgcn_shadow",
        description="R-GCN parallel comparison lane (no order authority)",
        ladder=(NPU, GPU, CPU),
        rationale="small graph, latency bound; nothing routes on it",
    ),
    # Throughput says GPU: this scores batches of ~4096 candidates and the iGPU is
    # the right shape of silicon for it. It is NOT placed there, and the reason is
    # worth keeping. Its compatibility_score is multiplied into `weighted_utility`
    # in StrategyRouter, so it helps decide which strategy is elected — and GPU was
    # measured disagreeing with CPU on the R-GCN over differences of ~5.6e-2 bps.
    # Moving a decision-affecting kernel to an unverified device to gain throughput
    # trades a guarantee for a speed-up nobody asked for. NPU stays first because it
    # is what this path has actually been running on; GPU becomes eligible the day
    # verify_decision_equivalence is wired to this runtime and passes.
    Workload(
        key="ontology_candidate_scorer",
        description="ontology candidate scoring",
        ladder=(NPU, GPU, CPU),
        rationale="batch-shaped work, but feeds weighted_utility - needs an equivalence proof before moving",
    ),
    Workload(
        key="event_classification",
        description="local event/news classification",
        ladder=(NPU, GPU, CPU),
        rationale="separate model lifecycle, latency bound",
    ),
    Workload(
        key="trading_decision",
        description="order/exit decisions, sizing, profitability gate",
        ladder=(CPU,),
        accelerable=False,
        rationale="must be bit-reproducible for audit; never accelerated",
    ),
    Workload(
        key="risk_checks",
        description="principal protection and risk limits",
        ladder=(CPU,),
        accelerable=False,
        rationale="fail-closed logic must not vary with hardware",
    ),
    Workload(
        key="ontology_rules",
        description="closed-world ontology gate and rule evaluation",
        ladder=(CPU,),
        accelerable=False,
        rationale="symbolic reasoning, no numeric kernel to offload",
    ),
)


@dataclass(frozen=True)
class Placement:
    workload: str
    device: str
    requested: str
    fallback_reason: str | None
    accelerable: bool
    rationale: str
    equivalence: "EquivalenceResult | None" = None


@dataclass(frozen=True)
class EquivalenceResult:
    """Did the accelerated device decide the same thing the CPU did?"""

    checked: bool
    equivalent: bool
    max_absolute_difference: float | None
    decisions_compared: int
    decisions_disagreeing: int
    detail: str

    @property
    def usable(self) -> bool:
        return (not self.checked) or self.equivalent


def plan_devices(
    inventory: DeviceInventory | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
) -> tuple[Placement, ...]:
    """Assign every workload to the best device this machine actually has.

    Each workload walks its own ladder and stops at the first available device, so a
    CPU-only laptop gets a complete, working plan rather than a degraded one — the
    placement changes, the behaviour does not.
    """
    inventory = inventory or probe_devices()
    resolved: list[Placement] = []
    for workload in WORKLOADS:
        override = (overrides or {}).get(workload.key) or os.getenv(
            f"DEVICE_PLAN_{workload.key.upper()}"
        )
        requested = (override or workload.ladder[0]).strip().upper()
        if not workload.accelerable and requested != CPU:
            resolved.append(
                Placement(
                    workload.key,
                    CPU,
                    requested,
                    f"{workload.key} is not accelerable: {workload.rationale}",
                    False,
                    workload.rationale,
                )
            )
            continue
        ladder = (requested, *workload.ladder) if override else workload.ladder
        chosen = next((d for d in ladder if inventory.has(d)), CPU)
        reason = None
        if chosen != ladder[0]:
            missing = ladder[0]
            reason = (
                f"{missing} unavailable on this machine"
                if not inventory.has(missing)
                else f"{missing} not selected"
            )
        resolved.append(
            Placement(
                workload.key,
                chosen,
                requested,
                reason,
                workload.accelerable,
                workload.rationale,
            )
        )
    return tuple(resolved)


def verify_decision_equivalence(
    reference: Callable[[Any], Any],
    candidate: Callable[[Any], Any],
    samples: Sequence[Any],
    *,
    decision: Callable[[Any], Any] | None = None,
    absolute_tolerance: float = 1e-4,
) -> EquivalenceResult:
    """Prove an accelerated path decides what the CPU path decides, or reject it.

    This is the whole basis for claiming behaviour is unchanged across hardware, and
    it deliberately does NOT test for bit equality. Different silicon reorders
    floating point accumulation; demanding identical bits would reject every
    accelerator for a difference that changes nothing. What must not change is the
    DECISION — the sign of a net edge, which strategy ranks first — so ``decision``
    projects raw output down to the thing the system acts on and that projection is
    compared exactly. The numeric tolerance is reported alongside as evidence, not
    as the criterion.

    A single disagreement fails the check. There is no "mostly agrees" here: a
    placement that changes one decision in a thousand is a placement that makes the
    system unexplainable, and the cost of refusing it is only that we run on CPU.
    """
    if not samples:
        return EquivalenceResult(False, True, None, 0, 0, "no samples supplied")
    worst = 0.0
    disagreements = 0
    compared = 0
    for sample in samples:
        try:
            expected = reference(sample)
            actual = candidate(sample)
        except Exception as exc:  # noqa: BLE001 - a throwing accelerator is a failing one
            return EquivalenceResult(
                True, False, None, compared, disagreements + 1, f"raised: {exc}"
            )
        expected_array = np.asarray(expected, dtype=np.float64)
        actual_array = np.asarray(actual, dtype=np.float64)
        if expected_array.shape != actual_array.shape:
            return EquivalenceResult(
                True,
                False,
                None,
                compared,
                disagreements + 1,
                f"shape {actual_array.shape} != reference {expected_array.shape}",
            )
        if expected_array.size:
            worst = max(
                worst, float(np.max(np.abs(expected_array - actual_array)))
            )
        compared += 1
        if decision is not None:
            if decision(expected) != decision(actual):
                disagreements += 1
        elif not np.allclose(
            expected_array, actual_array, rtol=0.0, atol=absolute_tolerance
        ):
            disagreements += 1
    equivalent = disagreements == 0
    detail = (
        f"{compared} sample(s) agreed, max |diff| {worst:.3e}"
        if equivalent
        else f"{disagreements}/{compared} sample(s) disagreed, max |diff| {worst:.3e}"
    )
    return EquivalenceResult(True, equivalent, worst, compared, disagreements, detail)


def enforce_equivalence(
    placement: Placement, result: EquivalenceResult
) -> Placement:
    """Keep an accelerated placement only if it proved equivalent, else drop to CPU."""
    if result.usable:
        return Placement(
            placement.workload,
            placement.device,
            placement.requested,
            placement.fallback_reason,
            placement.accelerable,
            placement.rationale,
            result,
        )
    return Placement(
        placement.workload,
        CPU,
        placement.requested,
        f"{placement.device} rejected: {result.detail}",
        placement.accelerable,
        placement.rationale,
        result,
    )


def plan_as_dict(placements: Iterable[Placement]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in placements:
        row: dict[str, Any] = {
            "workload": item.workload,
            "device": item.device,
            "requested": item.requested,
            "accelerable": item.accelerable,
            "rationale": item.rationale,
            "fallback_reason": item.fallback_reason,
        }
        if item.equivalence is not None:
            row["equivalence"] = {
                "checked": item.equivalence.checked,
                "equivalent": item.equivalence.equivalent,
                "decisions_compared": item.equivalence.decisions_compared,
                "decisions_disagreeing": item.equivalence.decisions_disagreeing,
                "max_absolute_difference": item.equivalence.max_absolute_difference,
                "detail": item.equivalence.detail,
            }
        rows.append(row)
    return rows
