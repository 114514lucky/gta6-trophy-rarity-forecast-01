"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Method: N-space dilution + progression.
--------------------------------------
An Exophase earn rate is a ratio, r_i(t) = E_i(t) / N(t): earners of
trophy i over players tracked. The two move for different reasons, so
the program models them separately instead of extrapolating the ratio:

* N(t), players tracked, grows as new players sync their profiles.
  It is fitted on the case history with two growth forms (a power law
  in log-time and a saturating exponential) and extrapolated to the
  target; the disagreement between the forms plus a scale factor
  gives N scenarios.
* E_i(t), earners of trophy i, grows from two drivers, both fitted
  per trophy: new joiners (a line in N, E_i = a_i + g_i * N, fitted
  over the last 24 h of distinct reads, g_i shrunk toward a class
  prior built from the trophy's own description) and tracked
  non-holders earning over time (a daily hazard h_i, fitted from
  the history after removing the joiner-driven part, also shrunk
  to its class prior). The forecast applies the joiner part first,
  then the hazard to the remaining non-holders over the horizon.
  Either driver alone fails where the other dominates: a pure
  joiner model stalls when N saturates but play continues, and a
  pure time trend misses dilution when N jumps.
* The forecast for each trophy is the weighted-quantile set of the
  scenario rates, widened by an extra logit spread for model error.

Why this beats the house baselines: persistence assumes no change
over 4.5 days, and the curve baseline fits logit(rate) against
log-time, which confounds dilution (N growth) with progression
(players earning more). Separating the two is what the launch-night
refresh showed: at the first refresh N jumped 638 -> 3,868 while
earners barely moved, so most rates fell several-fold even though
nobody "lost" a trophy.

Stdlib only, deterministic, no network, no wall clock, no
randomness. Reads every *.in in the input dir, writes <id>.out.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# sibling module would import (none is shipped; this stays one file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Extra logit spread convolved onto the scenario quantiles: covers
# model error (N-form miss, hazard drift, refresh lumpiness).
SIGMA_EXTRA = 0.6


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (same as the kit starter)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def days_since_release(when):
    return (when - RELEASE).total_seconds() / 86400.0


# ------------------------------------------------------------------
# Trophy classes: (marginal rate g, daily hazard h) priors from the
# trophy's own description. g = share of newly tracked players who
# hold the trophy; h = per-day chance a tracked non-holder earns it.
# Story chapters decay geometrically with depth (later chapters need
# many more story hours); grind/collectibles sit near zero; easy
# early activities sit high. The history fit overrides these wherever
# the case actually shows movement; they only anchor flat trophies.
# ------------------------------------------------------------------

def classify(trophy):
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or ""
    if "gone fishing" in desc:
        return (0.999, 0.0, "universal")
    if tier == "platinum":
        return (0.0002, 0.00002, "platinum")
    if desc.startswith("complete chapter"):
        try:
            n = int(desc.split("chapter")[1].strip().rstrip("."))
        except (ValueError, IndexError):
            n = 9
        # Calibrated on the day-2 snapshot (Nov 20 16:02 UTC, live
        # Exophase fetch): the marginal rate among newly tracked
        # players, (E_day2 - E_launch) / (N_day2 - N_launch), was
        # ~0.12 for Ch1 and ~0.017 for Ch2, and ~0.0001 for Ch3+;
        # priors sit between the day-2 value and a mature value,
        # since players keep progressing toward the target.
        # hazard h is anchored the same way: -ln(1 - r_day2) / 4,
        # attributing about half the day-2 accumulation to hazard.
        table = {1: (0.13, 0.032), 2: (0.02, 0.004), 3: (0.001, 0.0001),
                 4: (0.0005, 0.0001)}
        return table.get(n, (0.0004, 0.0001)) + ("story-chapter",)
    if "complete the final mission." in desc or "final mission." in desc:
        return (0.0005, 0.0001, "story-final")
    if "complete the final mission for" in desc:
        return (0.0004, 0.0002, "story-side-final")
    # Story missions in list order run from an early mission just
    # after the opening chapters to late-game operations, so the
    # marginal rate decays with story depth. Breakfast Run is the
    # exception: day-2 evidence puts its marginal rate near 0.04.
    if "breakfast run" in desc:
        return (0.03, 0.011, "story-mission-early")
    if "cartel heist" in desc:
        return (0.0008, 0.0002, "story-mission")
    if "child's play" in desc:
        return (0.0005, 0.0002, "story-mission")
    if "headliners" in desc:
        return (0.0004, 0.0002, "story-mission")
    if "vcia takedown" in desc:
        return (0.0004, 0.0002, "story-mission")
    if "operation savage glory" in desc:
        return (0.0004, 0.0002, "story-mission")
    grind = ("250 stunts" in desc or "100% completion" in desc
             or "shooting range" in desc or "every location" in desc
             or "20 different races" in desc or "8 mission strands" in desc
             or "collectible card sets" in desc or "100 locations" in desc
             or "50 challenges" in desc or "every spot" in desc
             or "5 bank vaults" in desc or "garage properties" in desc
             or "laundering" in desc or "scooter bros" in desc
             or "armored truck" in desc or "each gang" in desc
             or "explore all of leonida" in desc or "vehicle showroom" in desc
             or "mass appeal" in desc or "choices made" in desc
             or "second time's the charm" in desc)
    if grind:
        return (0.0001, 0.00003, "grind")
    if "fishing challenge" in desc:
        return (0.13, 0.036, "activity-easy")
    if "arcade game" in desc:
        return (0.015, 0.004, "activity")
    if "bowling alley" in desc:
        return (0.012, 0.003, "activity")
    if "discover 100 vehicles" in desc:
        return (0.017, 0.004, "activity")
    if "cocktail" in desc:
        return (0.0015, 0.0005, "activity")
    if "what the hell" in desc:
        return (0.007, 0.002, "activity")
    if "wanted level" in desc:
        return (0.003, 0.0008, "activity")
    if "try each type of activity" in desc:
        return (0.006, 0.001, "activity")
    if "rob 3 shops" in desc:
        return (0.0003, 0.0002, "mid")
    if "combat focus" in desc:
        return (0.0006, 0.0004, "mid")
    if "unscheduled departure" in desc:
        return (0.002, 0.001, "mid")
    if "rhinos" in desc:
        return (0.0018, 0.001, "mid")
    if "jewelry store" in desc:
        return (0.0003, 0.0002, "mid")
    if "unique animal" in desc:
        return (0.0001, 0.00003, "mid")
    if "spend $100,000" in desc:
        return (0.0003, 0.0002, "mid")
    if "melee drive-by" in desc:
        return (0.0004, 0.0003, "mid")
    return (0.0005, 0.0005, "other")


# ------------------------------------------------------------------
# History handling
# ------------------------------------------------------------------

def distinct_reads(history):
    """Collapse consecutive identical reads (Exophase refresh steps):
    keep a read when (players_tracked, rates) differs from the last
    kept one. Always keeps the first and the last read."""
    kept = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or ()))
        if key != prev:
            kept.append(read)
            prev = key
    return kept


def last_rate(history, idx):
    """Most recent non-null rate for trophy idx, else 0.0."""
    for read in reversed(history):
        rates = read.get("rates") or ()
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx]
    return 0.0


# ------------------------------------------------------------------
# N(t): players tracked at the target, as scenarios (value, weight)
# ------------------------------------------------------------------

def _ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None, None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    sse = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
    return slope, inter, sse


def fit_n_scenarios(reads, t_target_days, horizon_days):
    """Return [(N_target, weight)]. Two growth forms + scale spread;
    falls back to refresh-step multiples when N barely moved."""
    pts = []
    for read in reads:
        n = read.get("players_tracked")
        if n is None or n <= 0:
            continue
        t = max(0.0, days_since_release(parse_time(read["at"])))
        pts.append((t, float(n)))
    if not pts:
        return [(1000.0, 1.0)]
    n_last = pts[-1][1]
    n_first = pts[0][1]
    preds = []
    if len(pts) >= 3 and n_last > n_first * 1.05:
        # Power law: ln N = a + b ln(1 + t).
        xs = [math.log1p(t) for t, _ in pts]
        ys = [math.log(n) for _, n in pts]
        slope, inter, _ = _ols(xs, ys)
        if slope is not None and slope > 0:
            preds.append(math.exp(inter + slope * math.log1p(t_target_days)))
        # Saturating exponential: N = Ninf (1 - exp(-t / tau)),
        # tau by grid search on log-space error.
        best = None
        for tau in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0):
            terms = []
            ok = True
            for t, n in pts:
                f = 1.0 - math.exp(-t / tau)
                if f <= 1e-9:
                    ok = False
                    break
                terms.append(math.log(n) - math.log(f))
            if not ok:
                continue
            log_ninf = sum(terms) / len(terms)
            sse = sum((math.log(n) - (log_ninf + math.log(1.0 - math.exp(-t / tau)))) ** 2
                      for t, n in pts)
            if best is None or sse < best[0]:
                best = (sse, math.exp(log_ninf), tau)
        if best is not None:
            _, ninf, tau = best
            preds.append(ninf * (1.0 - math.exp(-t_target_days / tau)))
    if not preds:
        # Flat history: Exophase refreshes in steps; the first refresh
        # multiplied tracked players ~6x, and the day-2 snapshot
        # implies ~50x over 40 h. Longer horizons span more refreshes
        # plus organic growth, so the upper multiples grow with the
        # horizon. A long horizon with no refresh at all is
        # near-impossible (refreshes come hours apart), so the
        # no-refresh scenario's weight decays with the horizon.
        grow = 1.0 + horizon_days
        w_lo = 0.25 * math.exp(-horizon_days / 0.5)
        rest = 1.0 - w_lo
        base = n_last
        return [(base * 1.5, w_lo),
                (base * 6.0 * grow, rest * 0.33),
                (base * 25.0 * grow, rest * 0.40),
                (base * 100.0 * grow, rest * 0.27)]
    base = math.exp(sum(math.log(p) for p in preds) / len(preds))
    base = max(base, n_last)
    cap = n_last * 20.0
    out = []
    for scale, w in ((0.7, 0.25), (1.0, 0.5), (1.45, 0.25)):
        out.append((min(base * scale, cap), w))
    return out


# ------------------------------------------------------------------
# Per-trophy earners: E = a + g * N, fitted on the last 24 h
# ------------------------------------------------------------------

def fit_g(reads, idx, g_prior, t_checkpoint_days):
    """Return (a, g_base, g_lo, g_hi, identified). a is the intercept
    of the E-on-N line; with no signal the line passes through the
    last point with the prior slope."""
    window = []
    for read in reads:
        t = days_since_release(parse_time(read["at"]))
        if t < t_checkpoint_days - 1.0:
            continue
        rates = read.get("rates") or ()
        n = read.get("players_tracked")
        if n is None or n <= 0 or idx >= len(rates):
            continue
        r = rates[idx]
        if r is None:
            continue
        window.append((float(n), r / 100.0 * float(n)))
    if len(window) < 3:
        window = []
        for read in reads:
            rates = read.get("rates") or ()
            n = read.get("players_tracked")
            if n is None or n <= 0 or idx >= len(rates):
                continue
            r = rates[idx]
            if r is None:
                continue
            window.append((float(n), r / 100.0 * float(n)))
    if not window:
        return (0.0, g_prior, g_prior, g_prior, False)
    n_last, e_last = window[-1]
    ns = [w[0] for w in window]
    if len(window) >= 3 and max(ns) > min(ns) * 1.02:
        slope, inter, sse = _ols(ns, [w[1] for w in window])
        if slope is not None:
            dof = max(1, len(window) - 2)
            sigma2 = sse / dof
            sxx = sum((x - sum(ns) / len(ns)) ** 2 for x in ns)
            se = math.sqrt(sigma2 / sxx) if sxx > 0 else 0.5
            mean_e = sum(w[1] for w in window) / len(window)
            sst = sum((w[1] - mean_e) ** 2 for w in window)
            r2 = 1.0 - sse / sst if sst > 0 else 0.0
            wgt = len(window) / (len(window) + 4.0) if r2 > 0.5 else \
                len(window) / (len(window) + 10.0)
            g_fit = min(1.0, max(0.0, slope))
            g_base = wgt * g_fit + (1.0 - wgt) * g_prior
            a = inter if 0.0 <= inter <= e_last else e_last - g_base * n_last
            spread = 1.5 * se + 0.3 * abs(g_fit - g_prior)
            return (a, g_base,
                    max(0.0, g_base - spread),
                    min(1.0, g_base + spread), True)
    a = e_last - g_prior * n_last
    # Unidentified slope: the prior is the forecast, with a spread
    # that respects its scale -- near 1 (a trophy every new joiner
    # already holds) the downside is bounded by (1 - g), not by half
    # of g; near 0 the spread is proportionally wide.
    down = min(0.5 * g_prior, 5.0 * (1.0 - g_prior) + 0.01)
    up = max(0.5 * g_prior, 0.0005)
    return (a, g_prior, max(0.0, g_prior - down),
            min(1.0, g_prior + up), False)


def fit_h(reads, idx, g_base, h_prior, t_checkpoint_days):
    """Hazard for tracked non-holders: the per-day chance one earns
    the trophy. Estimated from the history after removing the
    joiner-driven part: over consecutive distinct reads,
    residual = dE - g * dN, and h = sum(residual) /
    sum((N - E) * dt). Shrunk halfway to the class prior; the prior
    alone when the history shows no movement."""
    pts = []
    for read in reads:
        t = days_since_release(parse_time(read["at"]))
        rates = read.get("rates") or ()
        n = read.get("players_tracked")
        if n is None or n <= 0 or idx >= len(rates):
            continue
        r = rates[idx]
        if r is None:
            continue
        pts.append((t, float(n), r / 100.0 * float(n)))
    if len(pts) < 3:
        # No movement to separate the hazard from the joiner rate:
        # the class prior g already represents total marginal growth
        # (it is calibrated to the day-2 outcome, hazard included),
        # so adding a prior hazard on top would double-count.
        return 0.0
    num = 0.0
    den = 0.0
    pairs = 0
    for (t0, n0, e0), (t1, n1, e1) in zip(pts, pts[1:]):
        dt = t1 - t0
        if dt <= 0 or n0 <= 0:
            continue
        residual = (e1 - e0) - g_base * (n1 - n0)
        pool = (n0 - e0) * dt
        if pool <= 0:
            continue
        num += residual
        den += pool
        pairs += 1
    if pairs < 2 or den <= 0:
        return 0.0
    h_fit = num / den
    if h_fit < 0:
        h_fit = 0.0
    if h_fit > 0.5:
        h_fit = 0.5
    return 0.5 * h_fit + 0.5 * h_prior


# ------------------------------------------------------------------
# Forecast
# ------------------------------------------------------------------

def weighted_quantile(points, levels):
    """points: [(value, weight)] -> one value per level."""
    pts = sorted(points)
    total = sum(w for _, w in pts)
    if total <= 0:
        return [pts[len(pts) // 2][0]] * len(levels)
    out = []
    cum = 0.0
    j = 0
    for lv in levels:
        target = lv * total
        while j < len(pts) - 1 and cum + pts[j][1] < target:
            cum += pts[j][1]
            j += 1
        out.append(pts[j][0])
    return out


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("levels are not the contest levels")
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    if not history or not trophies:
        raise ValueError("case has no history or no trophies")
    horizon_days = case["horizon_minutes"] / 1440.0
    checkpoint = parse_time(case["at"])
    t_check = days_since_release(checkpoint)
    t_target = t_check + horizon_days
    reads = distinct_reads(history)
    n_scen = fit_n_scenarios(reads, t_target, horizon_days)
    n_last = None
    for read in reversed(reads):
        if read.get("players_tracked"):
            n_last = float(read["players_tracked"])
            break
    if n_last is None:
        n_last = 1000.0
    rows = []
    for idx, trophy in enumerate(trophies):
        r_last = last_rate(history, idx)
        e_last = r_last / 100.0 * n_last
        g_prior, h_prior, _label = classify(trophy)
        if r_last >= 99.99:
            g_prior = 0.999
        a, g_base, g_lo, g_hi, _ident = fit_g(reads, idx, g_prior, t_check)
        h_final = fit_h(reads, idx, g_base, h_prior, t_check)
        points = []
        for n_tgt, wn in n_scen:
            for g_s, wg in ((g_lo, 0.25), (g_base, 0.5), (g_hi, 0.25)):
                e_join = a + g_s * n_tgt
                e_join = min(max(e_join, e_last), n_tgt)
                for h_s, wh in ((h_final * 0.5, 0.25),
                                (h_final, 0.5),
                                (h_final * 2.0, 0.25)):
                    # Joiners first, then the hazard acts on the
                    # remaining non-holders over the horizon.
                    e = e_join + (n_tgt - e_join) * (
                        1.0 - math.exp(-h_s * horizon_days))
                    e = min(max(e, e_last), n_tgt)
                    points.append((100.0 * e / n_tgt, wn * wg * wh))
        # Convolve model error: each scenario rate becomes three
        # points, shifted by -/+ SIGMA_EXTRA in logit space. (Adding
        # the spread on top of the scenario quantiles instead would
        # double-count the tails.)
        spread_points = []
        for rate, w in points:
            lg = logit(rate)
            spread_points.append((inv_logit(lg - SIGMA_EXTRA), 0.25 * w))
            spread_points.append((rate, 0.5 * w))
            spread_points.append((inv_logit(lg + SIGMA_EXTRA), 0.25 * w))
        qs = weighted_quantile(spread_points, LEVELS)
        qs = sorted(qs)
        rows.append({"id": trophy["id"],
                     "q": [round4(min(100.0, max(0.01, v))) for v in qs]})
    return {"trophies": rows}


def fallback_forecast(case):
    """Last-resort valid output if the model raises: the starter's
    persistence spread around the last read (or 1% with a wide
    spread when there is no history at all). A valid-but-crude
    forecast always beats a missing one."""
    horizon_days = max(0.0, case.get("horizon_minutes", 0) / 1440.0)
    spread = 0.5 * math.sqrt(horizon_days) if horizon_days > 0 else 0.1
    history = case.get("history") or []
    rows = []
    for idx, trophy in enumerate(case.get("trophies") or []):
        rate = last_rate(history, idx) if history else 1.0
        lg = logit(rate if rate else 1.0)
        rows.append({"id": trophy["id"],
                     "q": [round4(min(100.0, max(0.01, inv_logit(lg + z * spread))))
                           for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
