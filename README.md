# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/114514lucky/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/114514lucky/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Write a program that reads the earn rate of every Grand Theft Auto VI trophy on Exophase as the house recorded it up to 2026-11-21 11:30 UTC and forecasts each trophy's rate at 2026-11-26 00:00 UTC, the percent of players who will have earned it by then, as five quantiles a trophy, for the lowest pinball loss on the logit scale.

## Method

An Exophase earn rate is a ratio, r = E / N (earners over players tracked), and the two move for different reasons, so the program models them separately instead of extrapolating the ratio:

- **N(t), players tracked**: fitted with a power law in log-time and a saturating exponential, extrapolated to the target; their disagreement plus a scale factor gives N scenarios.
- **E(t), earners**: two drivers per trophy — new joiners (a line in N, fitted over the last 24 h of distinct reads, shrunk to a class prior from the trophy's description) and tracked non-holders earning over time (a daily hazard fitted from the history after removing the joiner part, also shrunk to its class prior).
- **Quantiles**: weighted quantiles over the scenario rates, convolved with a logit spread for model error.

Class priors (marginal rate among new joiners, hazard) are anchored on the live Exophase snapshot of 2026-11-20 16:02 UTC, which showed the launch-night cohort was heavily selected: Chapter 1's marginal rate was ~0.12, Chapter 2 ~0.017, Chapter 3+ ~0.0001, while most trophies sat at the 0.01% floor as tracked players grew ~50x in 40 hours.

Validation (local, no server grade available): on the real launch-night → Nov-20 transition the program scores pinball 0.64 against the house persistence baseline's 2.88; on a five-regime synthetic out-of-distribution suite (base / weekend N surge / N stall / binge / slow play) it beats both house baselines in 4 of 5 regimes (mean ratio to the better baseline 0.79; slow-play regime 1.08x persistence).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, deterministic, no network. It reads every `*.in` in the input directory and writes `<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`) to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=4w93rrrc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
