# Pre-registration S1: switchback perturbation of organization levers

Status: DRAFT 2026-10-07, written from research.tex Experiment 5.9. Not signed off; not in the ledger until Enrico
approves the decisions in section 0. Nothing has run on a model.

## 0. Decisions needed before this is final

1. Outcome scale. Accumulation trends upward over a run, so on and off periods sit at different points of the trend
   and a level contrast is confounded by time. Proposed: the outcome is the per-generation change in population mean
   self-play (first difference), so the contrast is a rate-of-accumulation contrast. Alternative: levels with a
   linear detrend, which assumes the trend's shape.
2. Schedule. Proposed: blocks of 30 generations, 300 generations per run (5 on and 5 off), and counterbalanced
   starting phase across seeds (seed 0 starts on, seed 1 starts off, seed 2 starts on), so the time trend is balanced
   across lever states. Alternative: all seeds start on.
3. Burn-in. Proposed: drop the first 5 generations after every switch from the contrast (the mixing-time estimate
   from the pilot replaces 5 if larger). The bias bound is reported with and without burn-in.
4. Lever set. Proposed five, one run per lever: teaching (routing on/off), verification (selfplay(n=200) on/off),
   quarantine (withhold unverified text, with verification on), migration (rate 0.25 every 5 generations on/off),
   selection (keep_best_k versus shinka_weighted). Shrinkage selection as written in research.tex maps to
   shinka_weighted; confirm.
5. Provider, model and cap, as in G1.

## 1. Question

Which organization levers change the rate of accumulation in a running population, measured inside one long run by
switching the lever on and off on a fixed schedule? Edge item 3 (controlled perturbation of the organization over
long runs), at the lowest cost per contrast.

## 2. Setting

- Base organization: G1's `organized` condition (8 agents in two groups of 4, broadcast within group, selfplay(n=200)
  verification, replace_if_better, migration, corpus). One lever is toggled; every other lever is held at the base.
- One run per lever per seed: 300 generations after the warm start; 3 seeds for the pilot, R by power afterwards.
- Hosted model, same provider and seeds as G1; deals and warm starts shared with G1 (same experiment seed and
  warm-start set), so S1's first block is comparable to G1's organized condition.
- The lever schedule is a config field (`org.schedule`); the active lever state is logged every generation.

## 3. Estimand

Let D_s(g) = Y_s(g) - Y_s(g-1) be the per-generation change of population mean held-out self-play in seed s, and
L(g) in {on, off} the lever state at generation g. For a lever,

    theta = E_s[ mean of D_s(g) over on-generations - mean of D_s(g) over off-generations ],

where burn-in generations after each switch are excluded (decision 3). theta is the lever's effect on the rate of
accumulation. It is an interrupted-series estimand: the population carries its state across switches, so the
contrast is biased by carryover; W Theorem 15.5 bounds the bias of the switchback estimator by 4 M lambda (1 + t0),
with lambda = 1/30 the switch rate, t0 the mixing time of the population state, and M the maximum per-generation
outcome magnitude. t0 is estimated from the autocorrelation of D_s(g) (first lag below 1/e) and reported with the
bound beside every theta.

## 4. Design and inference

- Within-run contrast averaged over switches, then over seeds; bootstrap over seeds for the interval, IQM point
  estimate. Secondary: a block-permutation test of the sharp null (permute the on/off labels of whole 30-generation
  blocks within a run, respecting the counterbalanced design).
- Pilot: 3 seeds per lever at 90 generations (three blocks) on the hosted model, to measure the per-seed SD of
  theta and t0. Then R by the power rule at the minimum effect, capped at 6 seeds per lever.
- Second block on a second hosted model; acceptance needs both blocks.

## 5. Decision rules, fixed now

- Minimum effect: 0.05 points per generation (1.5 points per 30-generation block; research.tex says one point of
  mean self-play per lever, which on the first-difference scale is 1/30 per generation, below what the bias bound
  will allow; 0.05 is proposed instead and flagged as a decision).
- Credit a lever with accumulation if theta >= 0.05 with the 95% interval excluding 0, and theta exceeds the bias
  bound, in both blocks. A lever whose theta is positive but below the bias bound is reported as "not separable from
  carryover at this switch rate".
- Falsification: if no lever passes, the long-run perturbation design is reported as underpowered at this scale and
  the switch period is revisited (longer blocks lower lambda and the bound).

## 6. Stopping and spend

Fixed horizon; the spend file's cap stops every job before the next paid call; censored runs are reported as such.
Per run: 300 x 8 revise calls plus repairs and teach calls, about 2,600 calls, about 28 M tokens. Five levers x 3
seeds = 15 runs for the pilot block at 90 generations (about 12,000 calls), the full 300-generation block about
40,000 calls. Dollars from the provider's price page; the previous session estimated about $5 per 300-generation run.

## 7. Multiplicity and null calibration

Five confirmatory tests (one theta per lever) with p-values from the block-permutation test; BH at q = 0.1 across
the ledger. Null band from the null stub run with the same schedule (the null stub has no accumulation, so its theta
distribution is the null of the whole procedure, including the schedule).

## 8. Outputs

Per lever: the series of Y_s(g) with on and off periods shaded; theta with interval, t0 and the bias bound; the
permutation p-value. One summary figure of theta per lever with bounds. Ledger rows S1-teaching, S1-verification,
S1-quarantine, S1-migration, S1-selection.

## 9. Nearest prior work

Switchback designs and the mixing-time bias bound: W chapter 15 (Definition 15.2, Theorem 15.5). Lever perturbation
over long runs in LLM-agent populations was not found in the 2026-10-06 searches; phrase per novelty-notes.md after a
fresh search.
