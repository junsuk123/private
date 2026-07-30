from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StrategyUtilityModelConfig:
    batch_size: int
    time_steps: int
    max_nodes: int
    feature_dim: int
    relation_count: int
    strategy_count: int
    hidden_dim: int = 16
    seed: int = 7


@dataclass(frozen=True)
class StrategyUtilityOutput:
    probability_success: np.ndarray
    gross_return_bps: np.ndarray
    cost_bps: np.ndarray
    mae_bps: np.ndarray
    mfe_bps: np.ndarray
    fill_probability: np.ndarray
    holding_seconds: np.ndarray
    aleatoric_uncertainty: np.ndarray
    utility: np.ndarray
    no_trade_probability: np.ndarray


class FixedShapeStrategyUtilityModel:
    """Dense fixed-shape R-GCN plus causal temporal pooling for shadow inference."""

    def __init__(self, config: StrategyUtilityModelConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / max(1, config.feature_dim) ** 0.5
        self.relation_weights = rng.normal(
            0, scale, (config.relation_count, config.feature_dim, config.hidden_dim)
        ).astype(np.float32)
        self.self_weight = rng.normal(
            0, scale, (config.feature_dim, config.hidden_dim)
        ).astype(np.float32)
        self.strategy_heads = rng.normal(
            0,
            1 / max(1, config.hidden_dim) ** 0.5,
            (config.strategy_count, config.hidden_dim, 8),
        ).astype(np.float32)
        self.no_trade_head = rng.normal(
            0, 1 / max(1, config.hidden_dim) ** 0.5, (config.hidden_dim,)
        ).astype(np.float32)
        temporal = np.arange(1, config.time_steps + 1, dtype=np.float32)
        self.temporal_weights = temporal / temporal.sum()

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            config=np.asarray(
                [
                    self.config.batch_size,
                    self.config.time_steps,
                    self.config.max_nodes,
                    self.config.feature_dim,
                    self.config.relation_count,
                    self.config.strategy_count,
                    self.config.hidden_dim,
                    self.config.seed,
                ],
                dtype=np.int64,
            ),
            relation_weights=self.relation_weights,
            self_weight=self.self_weight,
            strategy_heads=self.strategy_heads,
            no_trade_head=self.no_trade_head,
            temporal_weights=self.temporal_weights,
        )
        return target

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> FixedShapeStrategyUtilityModel:
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            values = tuple(int(value) for value in data["config"].tolist())
            if len(values) != 8:
                raise ValueError("invalid strategy utility checkpoint config")
            model = cls(StrategyUtilityModelConfig(*values))
            for name in (
                "relation_weights",
                "self_weight",
                "strategy_heads",
                "no_trade_head",
                "temporal_weights",
            ):
                expected = getattr(model, name).shape
                value = np.asarray(data[name], dtype=np.float32)
                if value.shape != expected or not np.isfinite(value).all():
                    raise ValueError(
                        f"invalid strategy utility checkpoint tensor {name}: "
                        f"{value.shape} != {expected}"
                    )
                setattr(model, name, value.copy())
        return model

    def infer(
        self,
        x: np.ndarray,
        adjacency: np.ndarray,
        node_mask: np.ndarray,
        strategy_mask: np.ndarray,
    ) -> StrategyUtilityOutput:
        raw, no_trade_raw = self.infer_raw(x, adjacency, node_mask, strategy_mask)
        return output_from_raw(raw, no_trade_raw, node_mask, strategy_mask)

    def infer_raw(
        self,
        x: np.ndarray,
        adjacency: np.ndarray,
        node_mask: np.ndarray,
        strategy_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate(x, adjacency, node_mask, strategy_mask)
        # [B,T,R,N,N] @ [B,T,N,F] -> [B,T,R,N,F]
        messages = np.einsum("btrij,btjf->btrif", adjacency, x, optimize=True)
        relational = np.einsum(
            "btrnf,rfh->btnh", messages, self.relation_weights, optimize=True
        )
        self_part = np.einsum("btnf,fh->btnh", x, self.self_weight, optimize=True)
        hidden = np.maximum(relational + self_part, 0)
        hidden *= node_mask[:, :, :, None]
        temporal = np.einsum(
            "t,btnh->bnh", self.temporal_weights, hidden, optimize=True
        )
        raw = np.einsum(
            "bnh,shk->bnsk", temporal, self.strategy_heads, optimize=True
        )
        no_trade_raw = np.einsum("bnh,h->bn", temporal, self.no_trade_head)
        return raw, no_trade_raw

    def _validate(
        self,
        x: np.ndarray,
        adjacency: np.ndarray,
        node_mask: np.ndarray,
        strategy_mask: np.ndarray,
    ) -> None:
        c = self.config
        expected = {
            "x": (c.batch_size, c.time_steps, c.max_nodes, c.feature_dim),
            "adjacency": (
                c.batch_size,
                c.time_steps,
                c.relation_count,
                c.max_nodes,
                c.max_nodes,
            ),
            "node_mask": (c.batch_size, c.time_steps, c.max_nodes),
            "strategy_mask": (c.batch_size, c.max_nodes, c.strategy_count),
        }
        actual = {
            "x": x.shape,
            "adjacency": adjacency.shape,
            "node_mask": node_mask.shape,
            "strategy_mask": strategy_mask.shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} shape {actual[name]} != fixed shape {shape}")
        if not all(
            np.isfinite(value).all()
            for value in (x, adjacency, node_mask, strategy_mask)
        ):
            raise ValueError("model inputs must be finite")


def output_from_raw(
    raw: np.ndarray,
    no_trade_raw: np.ndarray,
    node_mask: np.ndarray,
    strategy_mask: np.ndarray,
) -> StrategyUtilityOutput:
        probability = _sigmoid(raw[..., 0])
        cost = _softplus(raw[..., 2]) * 10
        mae = _softplus(raw[..., 3]) * 15
        mfe = _softplus(raw[..., 4]) * 20
        # MAE/MFE are conditional loss/win magnitudes.  Averaging raw gross
        # returns across a heavily imbalanced trigger set makes every rare
        # profitable setup look negative even when the calibrated success
        # probability is high.  Use the hurdle-model expectation instead:
        # P(win)*E[net win] - P(loss)*E[net loss].
        net = probability * mfe - (1.0 - probability) * mae
        gross = cost + net
        fill = _sigmoid(raw[..., 5])
        holding = _softplus(raw[..., 6]) * 60
        uncertainty = _softplus(raw[..., 7])
        utility = (
            net
            - uncertainty
            + 0.1 * fill * mfe
        )
        valid = node_mask[:, -1, :, None] * strategy_mask
        utility = np.where(valid > 0, utility, -np.inf)
        no_trade = _sigmoid(no_trade_raw)
        no_trade = np.where(node_mask[:, -1, :] > 0, no_trade, 1.0)
        return StrategyUtilityOutput(
            probability_success=probability,
            gross_return_bps=gross,
            cost_bps=cost,
            mae_bps=mae,
            mfe_bps=mfe,
            fill_probability=fill,
            holding_seconds=holding,
            aleatoric_uncertainty=uncertainty,
            utility=utility,
            no_trade_probability=no_trade,
        )

def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30, 30)
    return 1 / (1 + np.exp(-clipped))


def _softplus(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30, 30)
    return np.log1p(np.exp(clipped))
