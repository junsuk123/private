from app.routing.candidate_ranker import CandidateObservation, rank_candidates


def test_ranker_prefers_risk_adjusted_momentum_and_keeps_components():
    ranked = rank_candidates(
        (
            CandidateObservation(
                symbol="GOOD", momentum_12_1=0.20, realized_volatility=0.01,
                relative_strength=0.15, long_trend_score=1.0, liquidity_score=0.9,
                regime_fit=0.8, ontology_fit=0.8, gnn_suitability=0.7,
            ),
            CandidateObservation(
                symbol="NOISY", momentum_12_1=0.22, realized_volatility=0.08,
                relative_strength=0.10, long_trend_score=0.0, liquidity_score=0.4,
                regime_fit=0.5, ontology_fit=0.5, gnn_suitability=0.5,
            ),
        )
    )
    assert ranked[0].symbol == "GOOD"
    assert ranked[0].rank == 1
    assert "risk_adjusted_momentum" in ranked[0].components


def test_missing_gnn_is_neutral_not_a_block():
    ranked = rank_candidates((CandidateObservation(symbol="NEW"),))
    assert ranked[0].score == 0.5
    assert "gnn_suitability" in ranked[0].missing
