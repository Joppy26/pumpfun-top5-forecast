#!/usr/bin/env python3
"""Pump.fun top-5 forecast (make.fun pumpfun-top5-forecast-01).

Reads the house's 10-minute marks of pump.fun coins at /input (or
$MAKEFUN_INPUT_DIR), forecasts the top 5 by market cap at each of the seven
answer reads in /input/task.json, and writes /output/forecast.json (or
$MAKEFUN_OUTPUT_DIR).

Model: Monte Carlo over log10 market caps (in SOL).
- Each known coin is a log-normal random walk from its last mark: drift
  estimated from its recent slope, shrunk toward the global drift, saturating
  over ~2.5 h (momentum fades); volatility from its recent returns; a crash
  hazard (a coin can drop hard) that grows with the horizon.
- A newcomer field: virtual coins born after the cut, drawn fresh at every
  read, whose top cap grows slowly with the horizon. Newcomers are the main
  reason the top 5 turns over: in the rehearsal, 4 of the 5 places at the
  last read were held by coins no history line names.
- The forecast is the average of the seven per-read distributions (one
  forecast is scored at every read, so the average is the right single
  forecast for a log score).
- Allocation per place: name the coins whose place probability beats the
  unnamed share's per-coin value, at their probability; fill the remaining
  rows (up to 99) with p=0 for the coins most likely to be on the answer
  chart, so the unnamed share is shared by as few coins as possible and a
  newcomer taking a place scores the whole of it. Sizes are named at their
  probability only (no p=0 rows: a size bin is always nameable, so the
  proper allocation is best there).
- Deterministic: fixed seed, sorted iteration, rounded probabilities.

Standard library only. No network, no wall clock, no files outside the
repository. Two runs on the same history write identical bytes.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
N_SIM = 4000
MAX_COINS = 250
MAX_NAMED_PER_PLACE = 99
TAU_HOURS = 0.5          # drift saturates over this long: a coin's slope is
                         # a 10-minute signal, and coins peak then decay
                         # (Claude grew steadily for 40 minutes yet fell
                         # out of the top 5 by the last practice read)
DRIFT_LO = -0.6          # per-hour drift caps
DRIFT_HI = 0.5
SIGMA_LO = 0.25          # per sqrt-hour volatility floor
SIGMA_HI = 0.6
SIGMA_CAP = 0.65         # total log-cap sd cap at any horizon (a coin's cap
                         # mean-reverts; it does not random-walk to the moon)
CRASH_HAZARD_PER_H = 0.025
CRASH_MEAN = 0.8         # log10 drop of a crash
CRASH_SD = 0.3
DECAY_BASE_PER_H = 0.04  # post-peak decay: after the drift saturates, a
DECAY_PER_DRIFT = 0.06   # coin fades (a pump peaks, then decays); faster
                         # for coins that pumped hardest
LOGCAP_FLOOR = 1.0       # a coin's log10 cap in SOL never reads below this
NEWCOMER_RATE_PER_H = 3.0   # contender newcomers arriving per hour
NEWCOMER_MAX = 12        # most virtual newcomers drawn at one read
NEWCOMER_SD = 0.35
NEWCOMER_SPACING = 0.15  # log10 gap between the j-th newcomers
NEWCOMER_GROWTH = 0.15   # saturating log10 rise of the top newcomer cap
NEWCOMER_TAU_H = 2.0     # ... over this many hours
SIZE_TEMPER = 0.10       # uniform mass mixed into the size distribution
P_FLOOR = 1e-4
ROUND_DP = 6


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(path):
    marks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    return marks


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1
    output_name = task["output"]
    reads = [parse_time(r) for r in task["reads"]]

    marks = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not marks:
        raise SystemExit("the history holds no ok mark")
    marks.sort(key=lambda m: m["at"])
    last_at = parse_time(marks[-1]["at"])
    last_rows = marks[-1]["rows"]
    if len(last_rows) < places:
        raise SystemExit("the last mark ranks fewer coins than the places")

    # ---- per-coin series -------------------------------------------------
    # mint -> list of (hours since t0, log10 cap SOL, rank, venue, complete)
    t0 = parse_time(marks[0].get("t0") or marks[0]["at"])
    series = {}
    history_mints = set()
    for m in marks:
        h = (parse_time(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            mint = r["mint"]
            history_mints.add(mint)
            cap_sol = int(r["market_cap_lamports"]) / 1e9
            lc = math.log10(cap_sol) if cap_sol > 0 else LOGCAP_FLOOR
            series.setdefault(mint, []).append(
                (h, lc, int(r["rank"]), r.get("venue", "curve"),
                 bool(r.get("complete", False)), r.get("created_at", "")))
    for obs in series.values():
        obs.sort()

    # ---- global drift / volatility ---------------------------------------
    slopes = []
    for obs in series.values():
        if len(obs) >= 2:
            (h0, c0, _, _, _, _), (h1, c1, _, _, _, _) = obs[0], obs[-1]
            if h1 > h0:
                slopes.append((c1 - c0) / (h1 - h0))
    slopes.sort()
    global_drift = slopes[len(slopes) // 2] if slopes else 0.0
    global_drift = max(DRIFT_LO, min(DRIFT_HI, global_drift))

    # ---- coin parameters --------------------------------------------------
    # Only coins the last mark ranks: a coin that fell out of the top 200
    # has a stale cap (it decayed since), and a faded coin rebounding to
    # the top 5 is far less likely than a fresh pump doing so.
    last_mints = {r["mint"] for r in last_rows}
    # (mint, last logcap, drift/hr, sigma/sqrt-h, crash hazard/hr, last rank)
    coins = []
    for mint, obs in series.items():
        if mint not in last_mints:
            continue
        h_last, lc_last, rank_last, venue, complete, _ = obs[-1]
        # recent slope over the last up to 3 steps, recency weighted
        num = 0.0
        den = 0.0
        for (h0, c0, _, _, _, _), (h1, c1, _, _, _, _) in zip(obs[-4:], obs[-3:]):
            if h1 > h0:
                w = 1.0 + (h1 - h_last)  # nearer the end, larger weight
                num += w * (c1 - c0) / (h1 - h0)
                den += w
        n_eff = min(len(obs), 6)
        if den > 0:
            slope = num / den
            drift = (n_eff / (n_eff + 1.5)) * slope + (1.5 / (n_eff + 1.5)) * global_drift
        else:
            drift = global_drift
        drift = max(DRIFT_LO, min(DRIFT_HI, drift))
        # volatility from step returns
        rets = []
        for (h0, c0, _, _, _, _), (h1, c1, _, _, _, _) in zip(obs, obs[1:]):
            if h1 > h0:
                rets.append((c1 - c0) / math.sqrt(h1 - h0))
        if len(rets) >= 2:
            mean_r = sum(rets) / len(rets)
            var = sum((x - mean_r) ** 2 for x in rets) / (len(rets) - 1)
            sigma = math.sqrt(var)
        elif rets:
            sigma = abs(rets[0])
        else:
            sigma = SIGMA_LO
        sigma = max(SIGMA_LO, min(SIGMA_HI, sigma))
        hazard = CRASH_HAZARD_PER_H
        if venue == "curve" and not complete:
            drift = min(drift, 0.3)
            hazard = CRASH_HAZARD_PER_H * 2.0
        coins.append((mint, lc_last, drift, sigma, hazard, rank_last))
    # keep the coins most likely to matter: largest last cap first
    coins.sort(key=lambda c: (-c[1], c[0]))
    coins = coins[:MAX_COINS]
    mints = [c[0] for c in coins]
    top1_last = coins[0][1] if coins else 3.0
    # chart cutoff: the log-cap of the 100th-ranked coin in the last mark.
    # The answer chart is the top 100 of ALL coins, not just the ones
    # simulated here, so a simulated coin counts as on the chart only if
    # its cap clears this cutoff too.
    chart_cutoff = coins[min(99, len(coins) - 1)][1] if coins else LOGCAP_FLOOR

    horizons = [(r - last_at).total_seconds() / 3600.0 for r in reads]

    # ---- Monte Carlo -------------------------------------------------------
    rng = random.Random(SEED)
    gauss = rng.gauss
    rand = rng.random
    place_counts = [[dict() for _ in range(places)] for _ in reads]
    size_counts = [[dict() for _ in range(places)] for _ in reads]
    chart_counts = [dict() for _ in reads]
    newcomer_counts = [[0] * places for _ in reads]

    for ri, h in enumerate(horizons):
        h = max(h, 0.0)
        sqrt_h = math.sqrt(h)
        drift_factor = TAU_HOURS * (1.0 - math.exp(-h / TAU_HOURS)) if h > 0 else 0.0
        # per-coin mean and sd at this read: the drift runs its course over
        # TAU_HOURS, then the coin fades at its post-peak decay rate
        means = []
        sds = []
        for (_, lc_last, drift, sigma, hazard, _) in coins:
            m = lc_last + drift * drift_factor
            if h > TAU_HOURS:
                decay = DECAY_BASE_PER_H + DECAY_PER_DRIFT * max(0.0, drift)
                m -= decay * (h - TAU_HOURS)
            means.append(m)
            sds.append(min(SIGMA_CAP, sigma * sqrt_h) if h > 0 else 0.0)
        hazards = [1.0 - math.exp(-hz * h) if h > 0 else 0.0
                   for (_, _, _, _, hz, _) in coins]
        # newcomer top-cap mean at this read; the count of contender
        # newcomers grows with the horizon (few can arrive and pump in
        # 10 minutes, many over 7-13 hours); the top cap rises fast at
        # first, then saturates as the market's attention does
        new_mu = top1_last + NEWCOMER_GROWTH * (1.0 - math.exp(-h / NEWCOMER_TAU_H)) if h > 0 else top1_last
        n_new = min(NEWCOMER_MAX, max(2, int(math.ceil(NEWCOMER_RATE_PER_H * h)))) if h > 0 else 2
        for _ in range(N_SIM):
            vals = []
            for idx in range(len(coins)):
                v = means[idx] + (sds[idx] * gauss(0.0, 1.0) if sds[idx] > 0 else 0.0)
                if hazards[idx] > 0.0 and rand() < hazards[idx]:
                    v -= abs(gauss(CRASH_MEAN, CRASH_SD))
                if v < LOGCAP_FLOOR:
                    v = LOGCAP_FLOOR
                vals.append((v, idx))
            for j in range(n_new):
                v = gauss(new_mu - NEWCOMER_SPACING * j, NEWCOMER_SD)
                if v < LOGCAP_FLOOR:
                    v = LOGCAP_FLOOR
                vals.append((v, -1 - j))
            vals.sort(key=lambda t: (-t[0], t[1]))
            for k in range(places):
                idx = vals[k][1]
                if idx >= 0:
                    d = place_counts[ri][k]
                    d[idx] = d.get(idx, 0) + 1
                else:
                    newcomer_counts[ri][k] += 1
                b = bin_of(vals[k][0], edges)
                d = size_counts[ri][k]
                d[b] = d.get(b, 0) + 1
            for v, idx in vals[:100]:
                if idx >= 0 and v >= chart_cutoff:
                    d = chart_counts[ri]
                    d[idx] = d.get(idx, 0) + 1

    # ---- average over reads ------------------------------------------------
    # P(coin i at place k), averaged over reads
    place_prob = [dict() for _ in range(places)]  # place -> mint -> p
    size_prob = [dict() for _ in range(places)]   # place -> bin -> p
    chart_p = {}  # mint -> P(on the answer chart), averaged over reads
    for k in range(places):
        for ri in range(len(reads)):
            for idx, c in place_counts[ri][k].items():
                mint = mints[idx]
                place_prob[k][mint] = place_prob[k].get(mint, 0.0) + c / N_SIM
        for mint in place_prob[k]:
            place_prob[k][mint] /= len(reads)
        for ri in range(len(reads)):
            for b, c in size_counts[ri][k].items():
                size_prob[k][b] = size_prob[k].get(b, 0.0) + c / N_SIM
        for b in size_prob[k]:
            size_prob[k][b] /= len(reads)
    for ri in range(len(reads)):
        for idx, c in chart_counts[ri].items():
            mint = mints[idx]
            chart_p[mint] = chart_p.get(mint, 0.0) + c / N_SIM
    for mint in chart_p:
        chart_p[mint] /= len(reads)
    new_prob = []
    for k in range(places):
        new_prob.append(sum(newcomer_counts[ri][k] for ri in range(len(reads)))
                        / (N_SIM * len(reads)))

    # the coins most likely to be on the answer chart: these are the ones
    # named at every place (positive p for the place candidates, p=0 for
    # the rest), so the unnamed share is shared by as few chart coins as
    # possible and a newcomer taking a place scores the whole of it
    chart_order = sorted(chart_p.items(), key=lambda kv: (-kv[1], kv[0]))
    chart_named = [mint for mint, _ in chart_order[:MAX_NAMED_PER_PLACE]]
    # expected number of chart coins left unnamed
    exp_unnamed = max(1.0, 100.0 - sum(chart_p.get(m, 0.0) for m in chart_named))

    # ---- allocation --------------------------------------------------------
    probabilities = []
    named_per_place = []
    for k in range(places):
        q = place_prob[k]
        # greedy: name a coin at its probability while it beats the
        # unnamed share's per-coin value (the unnamed share is split over
        # the chart coins left unnamed, newcomers among them)
        order = sorted(q.items(), key=lambda kv: (-kv[1], kv[0]))
        named = {}
        total = 0.0
        for mint, p in order:
            if p <= 0.0:
                continue
            per_unnamed = max(0.0, 1.0 - total) / exp_unnamed
            if p > per_unnamed:
                named[mint] = p
                total += p
            else:
                break
        # fill up to MAX_NAMED_PER_PLACE with p=0 for the coins most
        # likely to be on the answer chart
        for mint in chart_named:
            if len(named) >= MAX_NAMED_PER_PLACE:
                break
            if mint not in named:
                named[mint] = 0.0
        # round positive probabilities to 6 decimals; drop ones that
        # round to zero (they become unnamed instead)
        final = {}
        for mint, p in named.items():
            if p > 0:
                rp = math.floor(p * 1e6) / 1e6
                if rp > 0:
                    final[mint] = rp
            else:
                final[mint] = 0.0
        s = sum(final.values())
        if s > 1.0:  # rounding safety: trim the smallest positive row
            for mint in sorted(final, key=lambda m: final[m]):
                if final[mint] > 0:
                    del final[mint]
                    break
        named_per_place.append(final)
        for mint, p in sorted(final.items()):
            probabilities.append({"mint": mint, "place": k + 1, "p": p})

    # sizes: every bin is nameable, so the proper allocation is best, but
    # the model is uncertain in the tails: mix in a little uniform mass so
    # no bin ever scores the floor, and name every bin
    sizes = []
    for k in range(places):
        q = size_prob[k]
        final = {}
        for b in range(n_bins):
            p = (1.0 - SIZE_TEMPER) * q.get(b, 0.0) + SIZE_TEMPER / n_bins
            rp = math.floor(p * 1e6) / 1e6
            if rp > 0:
                final[b] = rp
        s = sum(final.values())
        if s > 1.0:
            for b in sorted(final, key=lambda b: final[b]):
                del final[b]
                break
        for b, p in sorted(final.items()):
            sizes.append({"place": k + 1, "bin": b, "p": p})

    # top5: the five coins most likely to be in the top 5, ordered by
    # expected place
    tot = {}
    exp_place = {}
    for k in range(places):
        for mint, p in place_prob[k].items():
            tot[mint] = tot.get(mint, 0.0) + p
            exp_place[mint] = exp_place.get(mint, 0.0) + (k + 1) * p
    ranked = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    ranked.sort(key=lambda kv: (exp_place[kv[0]] / kv[1], kv[0]))
    top5 = [mint for mint, _ in ranked]
    # a top5 mint must be named in the history (it is: all come from it)
    top5 = [m for m in top5 if m in history_mints][:places]
    while len(top5) < places:
        for mint in mints:
            if mint not in top5:
                top5.append(mint)
                break

    doc = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(doc) + "\n")


def bin_of(r, edges):
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) >> 1
        if edges[mid] <= r:
            lo = mid + 1
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    main()
