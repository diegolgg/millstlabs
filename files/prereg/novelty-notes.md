# Novelty notes (targeted arXiv checks; web search quota was exhausted so venues were not checked separately)

Rule: a claim may say "not found in the 2026-10-06 arXiv check" and must cite the nearest three works. Never "never done".

## Claim C (causal teacher credit by randomized delivery or exact replay)
Nearest: Yang et al. 2025, Shapley + process-reward credit for multi-LLM agents, conceptual only (arXiv 2511.10687);
Li et al. 2026, counterfactual replay to localize failures in agent DAGs (2608.29228); Tang et al. 2026, Aumann–Shapley
attribution at million-agent scale (2605.11404). Also Shapley-Coop (2506.07388), SHARP (2602.08335).
Verdict: message-level credit is a crowded area; the specific instrument (randomized Bernoulli delivery plus regression,
checked against exact seeded replay, as a teacher's causal effect on a paired-seed student) was not found combined.
Defensible phrasing: "a randomized-delivery / counterfactual-replay estimator of a teacher's causal effect on a
paired-seed student, extending published Shapley- and replay-based multi-agent credit with a lighter causal design."

## Claim D (breakdown point of execution-based verification against sabotaged messages)
Nearest: Liu et al. 2026, consensus collapse past a corrupted-agent majority and a token-level fix (2604.17139);
Lee et al. 2026, (F+1)-robustness conditions under Byzantine agents (2605.09076); MAPLE-Guard 2026, gates on shared
memory writes and promotion, attack-success rates but no fraction sweep (2608.00426). Also TAMAS (2511.05269),
AI Control (2312.06942).
Verdict: fraction thresholds exist for aggregation and consensus rules; verification-before-promotion gates exist;
no breakdown-point curve for execution-based verification in a learning population was found.
Defensible phrasing: "an empirical breakdown-point curve for execution-based verification-before-adoption, building on
Byzantine-robustness thresholds and memory-poisoning defenses that have not reported this curve."

## Claim A (Henrich-style copy loss and copy variance, code vs prose)
Nearest: Vallinder & Hughes 2024 (2412.10270); Perez et al. 2024 telephone-game attractors (2407.04503);
Perez et al. 2024 cultural-evolution framework (2403.08882). Also POLIS (2507.21166), iterated-learning bias
amplification (2404.04286), TerraLingua (2603.16910).
Verdict: transmission-chain methodology for LLMs exists; Henrich's parameterization and a code-vs-prose medium comparison
were not found.
Defensible phrasing: "the first measurement we could find of Henrich-style transmission fidelity comparing code and
prose as the medium, building on the LLM telephone-game and cultural-evolution line."
