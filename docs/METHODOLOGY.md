# Methodology

This project tests trading-strategy ideas ("families") against market history and tries hard not to
fool itself. Most ideas fail. The point of the machinery below is that when one does not fail, the
result means something. Each section names the module that implements it.

## 1. Register first, then test once

Every family is written into a catalog (`data/mining/catalog.json`, read by `src/mining/catalog.py`)
with its rule, instruments and parameters **before** it is scored. Each family module under
`src/mining/batch5/` and `src/mining/batch6/` keeps its own copy of those parameters (`PARAMETERS`),
and the run scripts that score these modules refuse a family whose copy differs from its catalog
entry (`run_batch5_mining.parameter_drift`).

Each tested family gets one permanent `first_test` row in an append-only trial ledger
(`trial_ledger.record_first_test`), which records its holdout window when it has one. A second call
for the same family is a no-op, so a recorded window cannot be moved after the fact. When that row
is written depends on the scoring construction (section 3 of this document describes both):

- Per-trade construction: the `first_test` row is written before the deflated Sharpe ratio is
  computed, and the outcome follows through `trial_ledger.record_outcome`.
- Account-versus-index construction: the family is scored first. Its outcome is written next,
  through `trial_ledger.record_outcome_if_not_scored`, which accepts only one outcome per family on
  this construction. The `first_test` row is written after the outcome, so a failed outcome write
  leaves no row behind (`abi1_driver.record_abi1_outcome`).

Both writers build each record through the same check (`trial_ledger._build_outcome_record`), which
applies the rules in `src/mining/evidence.py` on every write.

The catalog parameters are fingerprinted (`evidence.parameter_fingerprint`). When a family whose
test was deferred is retried, a change to its parameters or to its signal code is refused
(`ParameterDriftError`). Changing an idea means registering a new family, which counts as a new
trial. Nothing is deleted: a failed test stays in the ledger, and the ledger is where the trial
counts behind the bar (section 2 of this document) come from.

Why: if you try enough ideas, some will look good by luck. The only defence is to count every try,
including the ones you would rather forget.

## 2. The deflated Sharpe ratio bar

A family passes only if its deflated Sharpe ratio is at least 0.95 (`constants.DSR_PASS_THRESHOLD`),
computed by `src/mining/dsr.py` following Bailey and Lopez de Prado (2014), "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality".

- `probabilistic_sharpe_ratio` gives the probability that the true Sharpe ratio is above a
  benchmark value, given the measured Sharpe ratio, the number of observations, and the skew and
  fat tails of the returns.
- `expected_max_sharpe_ratio` gives the best Sharpe ratio you would expect from pure luck across
  N trials, given how widely Sharpe ratios spread from one trial to the next. That becomes the
  benchmark the family must beat.
- `deflated_sharpe_ratio` is the probabilistic Sharpe ratio measured against that benchmark.

The trial count N and the spread of Sharpe ratios set the bar. The two scoring constructions in
section 3 of this document take them from different places.

The account-versus-index construction does not read the ledger at scoring time. It uses a fixed,
calibrated package: a trial count (`constants.ABI1_N_TRIALS`, 80) and the spread of daily Sharpe
ratios across those trials (`constants.DAILY_SR_VARIANCE`). The two were set together from the
families tested when the package was fixed and must always be used together, because a smaller trial
count with the same spread loosens the bar. Updating the package as the ledger grows is a deliberate
recalibration step, not automatic, so families tested after it was set are scored against the same
bar. A larger trial count can only make the bar harder. The price of a fixed package: once the ledger
records more tested families than the package counts, a family on this construction is deflated for
fewer tries than were made, and with the spread unchanged its bar is easier than one set at the full
count.

The per-trade construction reads both inputs from the ledger at run time. Its trial count is never
below the number of distinct families the ledger records as tested, nor below an effective count of
the run's own candidates (`num_independent_trials`, which counts correlated candidates as fewer
independent ones). Its spread is `trial_ledger.pooled_sr_variance`, taken over the latest valid
per-trade score of each family in the ledger; until enough families qualify, the run's own
candidates or a fixed floor (`constants.DEFAULT_SR_VARIANCE_FLOOR`) stand in. Per-trade and daily
Sharpe ratios are on different scales, so that pool reads per-trade scores only.

## 3. Score the account against the index, not the trades

Scoring lives in `src/mining/account_relative.py`. For every trading session the account exists,
including sessions where it holds nothing, it records what the account made and what SPY made, and
takes the difference in log terms (`log_excess`). That one daily series is what gets scored
(`score_account`), with the session count, not the trade count, as the sample size. At least 252
sessions are required.

Why: in a rising market, a strategy can be right on every trade and still lose to simply holding the
index, because it sits in cash most of the time. Scoring trades, or scoring absolute returns, rewards
that. Scoring the whole account against the index does not. Log excess is used because it adds up to
the log of the final wealth ratio, so "positive on average" and "finished ahead" mean the same thing.

On sessions it holds nothing, the account earns a small fixed sweep rate on its cash
(`sweep_cash_daily`, at `constants.IDLE_SWEEP_APY`); that is what the drivers pass in
(`abi1_driver.evaluate_family`). For idle money parked in bills or a bond fund the intended input
is `parked_cash_daily`, which pays that yield only after its spread and the tax on interest, not the
gross bill rate (`daily_cash_rate`) on its own. That is a convention, not a check: no shipped driver
passes either series, and `account_daily_returns` and `evaluate` accept whatever cash series a direct
caller supplies. Where the tax model applies, tax
is charged on realised gains, net of earlier losses, at each trade's exit (`netted_tax_factors`), and
`account_daily_returns` puts the pieces together.

After a family clears the bar, four pass/fail risk checks (`risk_refusals`) look at the account's own
curve. Three of the limits are the benchmark's own values over the same sessions; the fourth is a
fixed drawdown cap (`ABSOLUTE_MAX_DRAWDOWN_CAP`). Risk is a constraint, never part of the score, so a
cautious strategy cannot win by taking almost no exposure. For a strategy that adds a small
position on top of an index holding, the checks can measure only the added part
(`account_basis="overlay_only_excess"`), and `src/mining/overlay_risk_report.py` reports both views.

**Two scoring constructions.** Everything above describes the account-versus-index construction
(`construction="ABI-1"` in the ledger), used by `scripts/run_abi1_mining.py` and the other run
scripts that record through `abi1_driver.record_abi1_outcome`. The code also includes a per-trade
construction (`construction="per_trade"`), used by `scripts/run_batch3_mining.py`,
`run_batch5_mining.py`, `run_day_trading_mining.py`, `run_swing_mining.py`,
`run_expanded_universe_mining.py` and `run_index_addition_mining.py`. It scores a family's in-sample
trades instead of its daily account: the deflated Sharpe ratio uses the per-trade Sharpe ratio with
the trade count as the sample size, and the trial count and spread come from the ledger (section 2
of this document). Its holdout is the six months up to the test date
(`constants.HOLDOUT_WINDOW_MONTHS`), recorded in the family's `first_test` row, and a family that
clears the bar must then show a positive net edge over that holdout
(`backtest_engine.holdout_passes`).

## 4. Holdout windows and point-in-time data

On the account-versus-index construction, the most recent three years of scored sessions
(`constants.ABI1_HOLDOUT_SESSIONS`) are set aside before the main score is computed
(`split_holdout`), so the deflated Sharpe ratio sees only the earlier sessions. The set-aside stretch is then scored on
its own: its daily excess must reach the same 0.95 level as a probabilistic Sharpe ratio against a
zero benchmark, with no deflation (`holdout_gate`). A score that only works on data it has already
seen does not count. The per-trade construction's six-month holdout is described in section 3 of
this document.

Data must be what was knowable at the time:

- Regime and signal paths are causal: each value uses data up to that day only.
- FRED economic series used by the scoring drivers are fetched through the keyed FRED interface and
  only for series whose history is not revised after publication
  (`external_data.fetch_fred_series_point_in_time`, limited to `FRED_VERIFIED_UNREVISED_SERIES`). The
  keyless public-graph download (`external_data.fetch_fred_series`, and the spike-fade study script's
  own download) carries no such guarantee, and no scoring driver uses it. Each FRED reading is treated
  as unknown until a fixed number of trading sessions after its reference date
  (`signals.lag_boolean_regime_to_availability`). The lag is chosen per series by the caller; it
  approximates the publication delay rather than reconstructing each release date.
- FINRA margin-debt statistics are fetched from FINRA's current workbook at run time
  (`external_data.fetch_finra_margin_debt`), with no record of values as first published. Each monthly
  reading counts as known a fixed number of trading sessions after its month-end
  (`signals.finra_margin_debt_growth_available_series`, with the lag set in the family's parameters).
  FINRA's own publication delay can run past that lag in some months, so the lag is not guaranteed to
  be conservative; `src/mining/batch5/margin_debt_growth_regime.py` says so.
- CFTC open-interest reports (`src/mining/cftc_data.py`) count as known six calendar days after each
  report's as-of date (`AVAILABILITY_LAG_DAYS`). Reports held back by the three federal-shutdown
  publication gaps count only from the date by which they were all public (`PUBLICATION_BLACKOUTS`).
  The CFTC offers no archive of reports as first published and serves corrected values, so the series
  is labelled as not verified to be unrevised (`POINT_IN_TIME_STATUS`), and the loader refuses a cache
  manifest that does not carry that label.
- Company filings are read as filed, by filing date (`src/mining/edgar_point_in_time.py`).
- History that starts long before the traded instrument existed is capped
  (`signals.cap_pre_inception_history`), so a long warm-up cannot hide the signal.
- When a data vendor is missing recent bars, a run can end its history at the last complete session
  with `--history-end` (`src/mining/history_cutoff.py`). The cut-off is opt-in, checked against
  every series, recorded with the result, and refused if it drops too much data.

## 5. The cost realism gate

In the per-trade day-trading and swing pipelines (`scripts/run_day_trading_mining.py`,
`scripts/run_swing_mining.py`), every short-hold candidate faces a hard gate before scoring
(`intraday_cost_gate.evaluate_cost_realism_gate`). The spread comes from measured bid-ask quotes,
never an estimate. The round-trip spread must be below a ceiling (`INTRADAY_SPREAD_CEILING_BPS`), and
the candidate's net edge in its in-sample backtest must be at least three times that spread
(`INTRADAY_EDGE_OVER_COST_MULTIPLE`). An edge that only exists before costs is not an edge.

Intraday families scored on the account-versus-index construction do not go through this gate. Their
five drivers (`scripts/run_intraday_rsi2_reversion_abi1.py`, `run_green_day_morning_pullback_abi1.py`,
`run_vwap_cross_decaying_conviction_abi1.py`, `run_cross_asset_divergence_fvg_abi1.py` and
`run_fvg_reversal_cascade_pooled_abi1.py`) each price every traded symbol, plus SPY, from a dated
bid-ask quotes file by calling `execution_costs_for`. That helper and its quote sanity limit
(`QUOTE_GATE_BPS`) are defined in the per-trade batch5 driver, `scripts/run_batch5_mining.py`, which
uses them for its own candidates; the intraday drivers import them from there. A quote that is
missing, flagged suspect, malformed or wider than that sanity limit makes the call raise an error,
which these drivers do not catch, so the run stops before anything is scored. The measured round-trip
cost goes into each driver's own intraday backtest, and half the SPY spread is passed to
`abi1_driver.evaluate_family` as the benchmark's entry cost; `abi1_driver` reads no quotes itself.
Beyond that quote sanity limit, this path applies no cost-realism ceiling
(`INTRADAY_SPREAD_CEILING_BPS`) and no edge-over-cost multiple. (In `run_batch5_mining.py` itself a
failing quote skips the candidate, or drops that one member of a pooled family, instead of stopping
the run.)

## 6. One trial per family, not per ticker

When one idea is tested on several instruments, it is scored once and counts as one trial. On the
per-trade construction the trades are pooled into a single result (`src/mining/pooling.py`:
`pool_trades`, `pooled_daily_returns_by_exit_day`; the day-trading and swing pipelines use the same
design in `intraday_holdout.pool_intraday_trades` and `pool_swing_daily_returns`). On the
account-versus-index construction the members are combined into one account curve
(`abi1_driver.build_pooled_curve`; the pooled intraday drivers use
`day_trading_abi1_bridge.build_pooled_day_trading_curve`).
Testing each ticker separately would let the best ticker win by luck while counting the other
tickers as free tries.

## 7. Near misses

`src/mining/near_miss.py` lists families that came close without passing: those whose deflated
Sharpe ratio landed within 0.05 of the bar (`find_near_misses`), and those stopped by the cost gate
that still showed a real positive edge (`find_cost_gate_near_misses`). These are candidates for a
better-designed follow-up, which would be registered and tested as a new family.

## 8. The leveraged-ETF regime research

A separate line of work asks when holding a 3x leveraged index fund beats holding the index itself.
Leverage only pays when average returns are large relative to volatility, so the research tries to
detect market regimes.

- `src/regime_signal.py` fits a two-state hidden Markov model (bull and bear) to daily returns
  (`fit_hmm`), and `src/hmm_guards.py` rejects fits that are not usable (`fit_is_sane`).
- `src/causal_backtest.py` refits the model on a fixed schedule using only past data
  (`causal_pbull_path`), producing a daily probability of being in the bull state that never looks
  ahead.
- `src/gate_engine.py` turns that probability into a position of cash, 1x or 3x
  (`exposure_states`). It moves at most one step per day and uses separate entry and exit levels
  (hysteresis), so it does not flip back and forth around a single threshold.
- `src/strategy_sweep.py` and `src/objective.py` search the gate settings with block-bootstrap
  resampling and keep only settings whose extra drawdown over the plain index, averaged over the
  worst 5% of resampled paths, does not exceed a set limit. Single paths can still be worse than
  that average. `src/sweep_cache.py` stores each sweep result under a fingerprint of its inputs
  and of a fixed list of the functions and constants the sweep runs (`compute_abi_digest`,
  `sweep_fingerprint`), so editing anything on that list recomputes the sweep. A change to code
  outside the list is not detected; bumping `SWEEP_CACHE_VERSION` by hand invalidates every cached
  result. The fingerprint also covers the Python, numpy, pandas and pyarrow versions, so upgrading
  any of them recomputes the sweep.
- The most recent stretch of history is held out from the search and used once, to check the chosen
  setting.

## What this document does not cover

Results are summarised briefly in the README. The trial ledger itself and per-family result logs are
not published.

The run scripts in `scripts/` are shipped as reference code, not as a one-command reproduction. Each
one reads inputs that are not in this repository: the trial ledger, the curated candidate lists and
run matrices (some pinned by hash), the bid-ask quote samples used to measure trading costs (one
small sample for the index-addition family ships with its event cache), and price history (fetched
from data vendors at run time or read from a local cache; vendor terms do not allow redistributing it).
Without those inputs, a script stops with an error, such as a missing file or a hash mismatch, rather
than producing a result. The test suite exercises the same code paths
with small synthetic inputs, and that is what runs from a clean clone.
