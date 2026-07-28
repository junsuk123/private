# Operational Ontology Boundary

Existing RDF/OWL/SHACL assets remain the semantic TBox and validation source. Trading authorization uses `app.ontology.operational_gate`, a separate closed-world, point-in-time snapshot.

Every `OperationalFact` contains value, observation time, valid interval, source, and confidence. Missing facts are not interpreted as false or true: a missing required fact hard-blocks the strategy. Stale, not-yet-valid, low-confidence, and failed boolean requirements produce deterministic reason codes.

The gate outputs:

- ontology snapshot and validity interval;
- allowed strategy IDs;
- blocked strategy reasons;
- soft compatibility scores;
- source-linked explanation paths.

This prevents OWL open-world absence from becoming trading permission. Golden tests cover missing facts, stale facts, valid facts, confidence, compatibility, and explanations.
