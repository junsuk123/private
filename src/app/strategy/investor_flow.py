from __future__ import annotations

from dataclasses import dataclass

from app.schemas.domain import InvestorFlowSnapshot, InvestorGroup, MarketSnapshot, OrderAction


DOMESTIC_MARKETS = {"KRX", "KOSPI", "KOSDAQ", "KONEX"}
EPSILON = 1e-9


@dataclass(frozen=True)
class InvestorFlowMetrics:
    foreign_imbalance: float
    institution_imbalance: float
    retail_imbalance: float
    program_imbalance: float
    total_imbalance: float
    informed_imbalance: float
    retail_absorption: float
    price_impact_lambda: float
    signed_impact_efficiency: float
    volume_pressure: float


@dataclass(frozen=True)
class InvestorFlowAssessment:
    dominant_group: InvestorGroup
    action_bias: OrderAction
    score_adjustment: float
    confidence_adjustment: float
    supporting_factors: tuple[str, ...]
    contradicting_factors: tuple[str, ...]
    risk_factors: tuple[str, ...]
    reasoning_summary: tuple[str, ...]
    metrics: InvestorFlowMetrics | None = None


def assess_domestic_investor_flow(market: MarketSnapshot) -> InvestorFlowAssessment:
    if not is_domestic_market(market) or market.investor_flow is None:
        return InvestorFlowAssessment(
            dominant_group=InvestorGroup.UNKNOWN,
            action_bias=OrderAction.HOLD,
            score_adjustment=0.0,
            confidence_adjustment=0.0,
            supporting_factors=(),
            contradicting_factors=(),
            risk_factors=(),
            reasoning_summary=(),
            metrics=None,
        )

    flow = market.investor_flow
    metrics = compute_investor_flow_metrics(market)
    dominant = _dominant_group(metrics)
    smart_money = _suspected_smart_money(flow, metrics)

    score = _flow_score(metrics)
    confidence = max(-0.08, min(0.08, score * 0.04))
    support: list[str] = []
    contradiction: list[str] = []
    risk: list[str] = []
    summary: list[str] = []

    if metrics.informed_imbalance >= 0.012:
        support.append("InformedOrderFlowImbalance")
        summary.append("Foreign and institutional order-flow imbalance is positive after trading-value normalization.")
    elif metrics.informed_imbalance <= -0.015:
        contradiction.append("InformedOrderFlowDistribution")
        summary.append("Foreign and institutional order-flow imbalance is negative after trading-value normalization.")

    if metrics.foreign_imbalance > 0.005 and metrics.institution_imbalance > 0.005 and flow.price_change_rate > 0:
        support.append("ForeignInstitutionJointBuying")
    if metrics.foreign_imbalance < -0.005 and metrics.institution_imbalance < -0.005:
        contradiction.append("ForeignInstitutionJointSelling")

    if metrics.retail_absorption >= 0.0015 and metrics.informed_imbalance > 0:
        support.append("RetailSupplyAbsorbedByInformedFlow")
        summary.append("Retail selling is absorbed by foreign or institutional demand.")
    elif metrics.retail_absorption >= 0.0015 and metrics.informed_imbalance < 0:
        contradiction.append("RetailDemandMeetsInformedSelling")
        summary.append("Retail buying is meeting foreign or institutional selling pressure.")
    elif metrics.retail_absorption <= -0.0015:
        contradiction.append("CrowdedSameDirectionFlow")
        summary.append("Retail and informed-flow proxies move in the same direction, raising crowding risk.")

    if metrics.signed_impact_efficiency >= 0.0007 and metrics.volume_pressure >= 1.0:
        support.append("OrderFlowPriceConfirmation")
    elif metrics.signed_impact_efficiency <= -0.0007:
        contradiction.append("OrderFlowPriceDivergence")

    if smart_money:
        dominant = InvestorGroup.SUSPECTED_SMART_MONEY
        if flow.price_change_rate > 0 and metrics.informed_imbalance >= 0:
            support.append("SuspectedSmartMoneyAccumulation")
            summary.append("Large unexplained/program flow with positive return is treated as suspected accumulation.")
        else:
            contradiction.append("SuspectedSmartMoneyDistribution")
            risk.append("OrderFlowDistributionRisk")
            summary.append("Large unexplained/program flow with weak return is treated as suspected distribution risk.")

    if abs(metrics.price_impact_lambda) >= 0.08 and abs(metrics.total_imbalance) < 0.015:
        risk.append("ThinLiquidityPriceImpactRisk")

    action_bias = _action_bias(score, risk)
    return InvestorFlowAssessment(
        dominant_group=dominant,
        action_bias=action_bias,
        score_adjustment=round(score, 4),
        confidence_adjustment=round(confidence, 4),
        supporting_factors=tuple(dict.fromkeys(support)),
        contradicting_factors=tuple(dict.fromkeys(contradiction)),
        risk_factors=tuple(dict.fromkeys(risk)),
        reasoning_summary=tuple(summary),
        metrics=metrics,
    )


def compute_investor_flow_metrics(market: MarketSnapshot) -> InvestorFlowMetrics:
    flow = market.investor_flow
    if flow is None:
        zero = 0.0
        return InvestorFlowMetrics(zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)

    trading_value = max(1.0, abs(flow.trading_value) or abs(market.average_daily_trading_value) or 1.0)
    foreign = flow.foreign_net_buy / trading_value
    institution = flow.institution_net_buy / trading_value
    retail = flow.retail_net_buy / trading_value
    program = flow.program_net_buy / trading_value
    total = foreign + institution + retail + program

    # Foreign and institutional flow are treated as informed-flow proxies; retail is useful but noisier.
    informed = 0.55 * foreign + 0.45 * institution - 0.20 * retail + 0.15 * program
    absorption = -(retail * (0.55 * foreign + 0.45 * institution))
    lambda_proxy = flow.price_change_rate / total if abs(total) > 0.002 else 0.0
    signed_efficiency = flow.price_change_rate * informed
    volume_pressure = max(0.0, flow.volume_change_rate)

    return InvestorFlowMetrics(
        foreign_imbalance=round(foreign, 6),
        institution_imbalance=round(institution, 6),
        retail_imbalance=round(retail, 6),
        program_imbalance=round(program, 6),
        total_imbalance=round(total, 6),
        informed_imbalance=round(informed, 6),
        retail_absorption=round(absorption, 6),
        price_impact_lambda=round(lambda_proxy, 6),
        signed_impact_efficiency=round(signed_efficiency, 6),
        volume_pressure=round(volume_pressure, 6),
    )


def is_domestic_market(market: MarketSnapshot) -> bool:
    market_name = market.market.upper()
    return market_name in DOMESTIC_MARKETS or market.ticker.endswith(".KS") or market.ticker.isdigit()


def _flow_score(metrics: InvestorFlowMetrics) -> float:
    raw = (
        16.0 * metrics.informed_imbalance
        + 10.0 * metrics.retail_absorption
        + 80.0 * metrics.signed_impact_efficiency
        + 0.10 * min(2.5, metrics.volume_pressure)
    )
    return max(-1.4, min(1.4, raw))


def _dominant_group(metrics: InvestorFlowMetrics) -> InvestorGroup:
    ranked = sorted(
        (
            (abs(metrics.foreign_imbalance), InvestorGroup.FOREIGN),
            (abs(metrics.institution_imbalance), InvestorGroup.INSTITUTION),
            (abs(metrics.retail_imbalance), InvestorGroup.RETAIL),
        ),
        reverse=True,
        key=lambda item: item[0],
    )
    if ranked[0][0] < 0.01:
        return InvestorGroup.MIXED
    return ranked[0][1]


def _suspected_smart_money(flow: InvestorFlowSnapshot, metrics: InvestorFlowMetrics) -> bool:
    known_gap = abs(metrics.foreign_imbalance + metrics.institution_imbalance + metrics.retail_imbalance)
    unexplained_or_program = known_gap < 0.01 or abs(metrics.program_imbalance) >= 0.012
    return metrics.volume_pressure >= 1.5 and unexplained_or_program


def _action_bias(score: float, risk_factors: list[str]) -> OrderAction:
    if risk_factors and score < 0.2:
        return OrderAction.REDUCE
    if score >= 0.6:
        return OrderAction.BUY
    if score <= -0.7:
        return OrderAction.SELL
    if score <= -0.3:
        return OrderAction.REDUCE
    return OrderAction.HOLD
