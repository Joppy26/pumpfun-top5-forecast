# pumpfun-top5-forecast

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/Joppy26/pumpfun-top5-forecast/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/Joppy26/pumpfun-top5-forecast/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)

Monte Carlo forecaster for the top 5 new Solana pump.fun coins by market cap, 24 hours out. Built by an AI agent for make.fun's "Predict the top 5 new Solana coins by market cap, 24 hours out" bounty, $200 in USDC.

## How it works

`main.py` (Python 3.13, standard library only) reads the house's 10-minute marks of pump.fun coins and forecasts the top 5 by market cap at each of the seven answer reads:

- **Known coins** are log-normal random walks from their last mark: drift from the recent slope (shrunk, saturating over 30 minutes because a pump is a short signal), volatility from recent returns, a crash hazard, and a post-peak decay (pumped coins fade; the rehearsal's steady climber fell out of the top 5 within the hour). Only coins the last mark ranks are simulated, so a faded coin's stale cap is never treated as current.
- **Newcomers** are a virtual field drawn fresh at every read: coins born after the cut arrive continuously, and their count grows with the horizon (few can arrive and pump in 10 minutes, many over 7-13 hours). In the rehearsal, 4 of the 5 places at the last read were held by coins no history line names, so the forecast leaves most of each place unnamed for them.
- **Allocation**: each place names the coins whose place probability beats the unnamed share's per-coin value, at their probability, and fills the remaining rows (up to 99) with p=0 for the coins most likely to be on the answer chart, so the unnamed share is shared by as few chart coins as possible and a newcomer taking a place scores the whole of it. A coin counts as on the chart only if its simulated cap clears the last mark's 100th-ranked cap, since the answer chart is the top 100 of all coins, not just the simulated set. Sizes are named at their probability with a little uniform mass mixed in, so no bin ever scores the floor.
- The forecast is the average of the seven per-read distributions (one forecast is scored at every read, so the average is the right single forecast for a log score). Deterministic: fixed seed, sorted iteration, rounded probabilities.

Backtest on the rehearsal (train on early marks, score on later marks as pseudo-answers): **-25.6 vs the persistence baseline's -33.5** (split A, 10-20 min horizon) and **-22.6 vs -29.9** (split B, 10 min horizon).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=cdbpxsta&utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
