"""Pre-registered constants for the strategy candidate-mining pipeline.

Frozen once, as the mining pipeline's pre-registration.
Changing any of these after the pipeline's first run requires the same
discipline the pipeline demands of what it screens: a documented, dated
reason, not a silent tune.
"""

# Deflated Sharpe Ratio pass bar (the DSR scoring stage). A DSR >= this
# value is required to reach the holdout. 0.95 is the paper's own illustrative
# "reasonably confident" threshold (Bailey & Lopez de Prado 2014, "A
# NUMERICAL EXAMPLE" section) -- adopted as the mining pipeline's
# pre-registered bar in the absence of a stronger domain-specific reason.
DSR_PASS_THRESHOLD = 0.95

# Holdout window length in months, pinned per family_id at that family's
# first-ever test (trial_ledger.py). 6 months for the per-trade holdout
# stage.
HOLDOUT_WINDOW_MONTHS = 6

# Candidate count target per run -- a soft target, not a hard cap. The
# catalog's actual eligible count after the run-time eligibility filter
# (ruled_out.filter_eligible) governs the real N tested in any given run.
CANDIDATE_COUNT_TARGET_MIN = 20
CANDIDATE_COUNT_TARGET_MAX = 40

# Token budget ceiling for one LLM-assisted mining run. A default the
# caller may override.
DEFAULT_BUDGET_CEILING_TOKENS = 300_000

# Short-term-capital-gains tax rate applied to taxable-account candidates
# (the cost/tax inputs stage). A tax-model helper would take `rate` as a
# REQUIRED caller-supplied argument with no internal rate source, so this is
# the mining pipeline's own pre-registered value: an assumed 12% federal
# ordinary-income bracket (Rev. Proc. 2025-32), a modelling assumption rather
# than a statement about any particular account.
STCG_TAX_RATE = 0.12

# Minimum in-sample trade count required before a candidate is scored at
# all (DSR scoring gate, pre-registered): below this, sample skew/kurtosis
# are too noisy for PSR's asymptotic-normality assumption to be credible
# (sample Pearson kurtosis is mechanically bounded near n for small n). A
# candidate with fewer in-sample trades than this is cut at the DSR scoring
# stage with stage="cut_dsr" and a reason noting insufficient trades, never
# silently scored anyway.
MIN_TRADES_FOR_SCORING = 15

# Minimum HOLDOUT trade count required before the holdout sign-check gate
# (backtest_engine.holdout_passes) may decide a family's fate. The
# holdout gate is a bare sign check with no minimum trade count, which lets a
# family pass on a single trade; many families rest on only a handful of
# holdout trades, so this is not a rare edge case.
#
# A family whose holdout falls below this floor is recorded
# "insufficient_evidence" (evidence.build_insufficient_holdout_trades),
# NOT "cut_holdout" and NOT "survived": the in-sample DSR bar was
# genuinely cleared, but the confirmatory look could not be trusted, so
# the family is deferred and stays eligible for a future look with more
# data -- exactly the treatment MIN_TRADES_FOR_SCORING already gives an
# underpowered IN-SAMPLE candidate, extended to the holdout side of the
# same gate.
#
# Fixed at 20 (a stricter 30 and a looser 10 were the alternatives).
MIN_HOLDOUT_TRADES = 20

# Fallback cross-candidate Sharpe-ratio variance (DSR scoring), used ONLY when
# a run has fewer than 2 candidates with a valid sharpe_in_sample to
# estimate variance empirically (e.g. a single-candidate run). Small and
# conservative rather than an ungrounded round figure -- pre-registering it
# here, even as an admittedly rough placeholder, is strictly better than an
# ad hoc number invented per run. Units: PER-TRADE Sharpe-ratio variance
# (matching sharpe_in_sample's own convention -- see dsr.py's frequency
# warning). It is a placeholder, not an empirically derived figure.
DEFAULT_SR_VARIANCE_FLOOR = 0.0004

# Minimum number of DISTINCT families' pooled sharpe_in_sample values
# (trial_ledger.pooled_sr_variance) needed before the pooled estimate is
# trusted over the per-run/floor fallback chain above. Below this, the
# tier-2/tier-3 logic still applies unchanged. Pooling is used because
# sr_variance, not n_trials, is the DOMINANT term in the DSR bar: it is a
# straight multiplier, while n_trials sits inside a slowly-growing factor.
# Fixed at 10. The pool only grows from records written going forward (no
# historical backfill, since earlier per-run result artifacts are not
# durable), so this constant matters only for how quickly a fragile small
# pool starts being trusted, never for whether pooling happens at all.
MIN_POOLED_SR_VARIANCE_SAMPLES = 10

# Cap on how much history a candidate's daily-return series (used ONLY
# for the DSR's cross-candidate correlation estimate) carries into its
# persisted artifact file. Embedding a candidate's FULL in-sample
# daily-return history (a decade or more from data_pipeline.fetch_ohlc's
# default 'max' lookback) makes the mining run's per-candidate payload
# balloon to hundreds of kilobytes per candidate. The correlation estimate
# does not need decades of mostly-zero daily observations to be meaningful;
# this bounds each candidate's *persisted correlation artifact* to its most
# recent N trading days (the backtest itself, and
# net_edge_bps/sharpe/skew/kurtosis, still use the candidate's FULL
# available in-sample history -- only the correlation-purpose daily series
# is windowed).
DAILY_RETURNS_CORRELATION_WINDOW_DAYS = 750

# Round-trip spread ceiling for intraday/swing candidates (the rough-backtest
# cost gate of the intraday/swing mining extension). Tighter than the daily
# pipeline's existing 300 bps sanity check. Liquid ETFs typically quote spreads of a
# basis point or two; thin small-caps quote spreads an order of magnitude
# worse, and short-hold strategies on them are not viable. 20 bps gives real
# margin above liquid-name spreads while structurally excluding that
# thin-liquidity profile.
INTRADAY_SPREAD_CEILING_BPS = 20.0

# Minimum ratio of a candidate's own measured net edge per round trip to
# its measured round-trip spread, required before a candidate can even
# reach real backtesting (the rough-backtest stage). Screens out "the edge
# is barely bigger than the cost" at the mining stage -- the specific failure mode
# of short-hold strategies. No formal derivation for 3x specifically
# (unlike DSR_PASS_THRESHOLD's paper citation or STCG_TAX_RATE's Rev. Proc.
# citation above) -- this is a judgment call, flagged as such rather than
# dressed up as derived.
INTRADAY_EDGE_OVER_COST_MULTIPLE = 3.0

# --- Evidence lifecycle ----------------------------------------------------

# Lifetime cap on how many EVIDENCE-RETRIES one family_id may ever receive
# through the deferred-retest path. A family recorded as
# "insufficient_evidence" is not disproven, so it is allowed to come back --
# but each return that clears the DSR bar is another look at the SAME pinned
# holdout window, and unbounded retries (widen to 5 instruments, then 8, then
# 12, until something passes) is precisely the fishing this state exists to
# prevent.
#
# JUDGMENT CALL, flagged as such rather than dressed up as derived -- there is
# no citation supporting 2 specifically, the same honesty the
# INTRADAY_EDGE_OVER_COST_MULTIPLE comment above applies to its own 3x. Two
# things make it defensible anyway:
#   (a) It is strictly TIGHTER than the status quo on the path it governs:
#       a merely-cut family may otherwise be re-tested without limit
#       (ruled_out.filter_eligible excludes only status != "eligible" or a
#       "survived" outcome).
#   (b) Reaching the cap has principled escapes that are not "give up":
#       record the family with reactivation kind "none_available" (an honest
#       terminal statement), or curate the mechanism as a NEW family_id, which
#       pays its own trial charge rather than smuggling extra looks onto an
#       old one.
# 1 (stricter) and 3 (looser) are both defensible; the value is simply the
# one number in this lifecycle that has no derivation of its own.
MAX_EVIDENCE_RETRIES_PER_FAMILY = 2

# --- Cross-sectional panel -------------------------------------------------

# The price basis every panel fetch and cache entry is stamped with. This
# RECORDS EXISTING BEHAVIOUR rather than changing anything:
# data_pipeline.fetch_ohlc has always passed auto_adjust=True. Naming it lets
# it enter the panel cache key and manifest, for the same reason
# intraday_data.py already puts adjustment in its own cache key -- two
# adjustment bases must never share a cache entry, because yfinance silently
# restates history and a topped-up cache would mix them inside one file with
# no error raised anywhere.
PANEL_PRICE_ADJUSTMENT = "yfinance_auto_adjust"

# How many member tickers may fail to fetch before a panel build refuses to
# proceed. ZERO, and the asymmetry with pooled testing is the whole reason:
# in a per-ticker pooled test a failed ticker costs that ticker's trades and
# nothing else, but in a CROSS-SECTIONAL breadth statistic a failed member
# corrupts every date's denominator -- a ratio with an unknown member is
# unknown, not approximate.
#
# JUDGMENT CALL, but not an aspirational one: a large universe fetch can
# complete with zero failures, and on a small-fund basis a re-fetch costs
# seconds. A floor already met in practice, and cheap to re-meet, is a real
# floor rather than an ideological one.
PANEL_MAX_FETCH_FAILURES = 0

# Minimum number of SEASONED members that must resolve on a given session for
# that session's breadth value to be considered known. DERIVED from the
# sector_spdr_11 basis's own founding cohort, and valid ONLY for that basis:
# exactly nine Select Sector SPDR funds carry continuous history from
# 1998-12-22 -- XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY -- while XLRE
# joins 2015-10-08 and XLC 2018-06-19. Nine therefore encodes "every fund
# that existed at the basis's inception must resolve", not a tolerance.
#
# A session resolving fewer than this yields NaN and is dropped rather than
# estimated. Boundary follows the house >=-passes convention (exactly 9
# passes), matching DSR_PASS_THRESHOLD and MIN_TRADES_FOR_SCORING above. A
# DIFFERENT membership basis must derive its own value from its own founding
# cohort -- this number is not portable.
PANEL_MIN_RESOLVED_MEMBERS = 9

# --- Near-miss candidate tracking ------------------------------------------

# How close a CUT family's own recorded DSR must be to DSR_PASS_THRESHOLD --
# an ABSOLUTE distance on DSR's own [0, 1] probability scale (DSR is a
# PSR/CDF value, not an open-ended magnitude like a Sharpe ratio, so an
# absolute margin is the natural unit here, unlike e.g. a percentage-of-bar
# margin which would need its own justification for why a fraction is
# meaningful on a probability) -- before it is flagged as a "near miss" for
# later revisit. The design holds the 0.95 DSR_PASS_THRESHOLD bar as-is, and
# compensates for how strict that is by (a) testing more candidates over time
# and (b) tracking near-misses so they can be revisited later (e.g. once the
# pooled sr_variance estimate above has more samples, once more historical
# data exists, or once the instrument universe widens) instead of being
# forgotten. This constant governs (b); see src/mining/near_miss.py.
#
# JUDGMENT CALL, flagged as such rather than dressed up as derived -- same
# convention as MAX_EVIDENCE_RETRIES_PER_FAMILY and
# INTRADAY_EDGE_OVER_COST_MULTIPLE above; there is no citation supporting
# 0.05 specifically. 0.05 (5 percentage points, i.e. DSR in [0.90, 0.95))
# was chosen as a round, easily-explained band. Tighter (e.g. 0.02) or looser
# (e.g. 0.10) are both defensible; this is not derived from any theory of
# how DSR estimation error scales.
NEAR_MISS_DSR_MARGIN = 0.05

# Ceiling on the fraction of sessions in a whole snapshot that may be BROKEN
# (at least one seasoned member missing a bar, so the breadth value there is
# genuinely unknown and is dropped rather than estimated) before a snapshot
# build refuses to write at all.
#
# JUDGMENT CALL -- no derivation exists for 0.001 specifically -- but
# empirically anchored rather than invented: a real eleven-fund panel built
# over a multi-decade span had only a negligible fraction of broken sessions,
# leaving a wide margin under this ceiling.
#
# Every broken date is named individually in the snapshot's provenance
# regardless of this fraction, so a human always sees the actual gaps and not
# merely a ratio. PATH-DEPENDENT indicators (a cumulative advance-decline line,
# an EWMA oscillator) ignore this constant entirely and require ZERO broken
# sessions inside their accumulated span, because pandas .cumsum()/.ewm() skip
# NaN and would silently bridge a gap instead of failing on it.
PANEL_MAX_BROKEN_SESSION_FRACTION = 0.001


# ---------------------------------------------------------------- ABI-1
# The account-relative ("ABI-1") construction scores the whole ACCOUNT
# against SPY on every trading session including idle ones, so its Sharpe
# is PER-SESSION. Everything below is in those units and must never be
# mixed with the per-trade constants above.

# Cross-candidate Sharpe-ratio variance for the DAILY construction.
# Fixed as a PACKAGE with ABI1_N_TRIALS below: the variance population and
# the trial population must be the same set, so this value and that trial
# count may never be used apart. Pairing this with a smaller trial count
# silently LOOSENS the bar.
#
# BASIS -- the null sampling bound: dsr.py's own PSR denominator already
# assumes var(SR-hat) = denom**2 / (n - 1), which collapses to ~1/n at Sharpe
# near zero, so the null-implied dispersion of a daily Sharpe estimated over
# n sessions is 1/n. Averaging that over the per-family session counts of
# ALL tested families, corrected for their heterogeneity, gives the value
# below. Every tested family counts -- including the pooled multi-instrument
# families that cannot themselves produce a single account curve -- because
# excluding a tested family is testing for free, the exact multiple-testing
# loophole the DSR exists to close.
#
# DO NOT source a daily sr_variance from `daily_moments.sr`: that series is
# built from `strategy_daily`, which never subtracts SPY, credits idle
# sessions 0.0 instead of the cash rate, and applies no tax. The field is
# already named "daily" and written into every run artifact, so it is the
# thing a future reader will pool by accident.
DAILY_SR_VARIANCE = 2.7829e-4

# Trial count paired with DAILY_SR_VARIANCE above. This is
# the count of families TESTED, not the count scoreable under ABI-1.
ABI1_N_TRIALS = 80

# Long-term capital-gains rate, applied to round trips held longer than
# LONG_TERM_HOLDING_DAYS. Trades held over a year can be a small share of
# trades yet carry a large share of the gains (especially for
# trend-following families, which hold their winners), so a single flat
# short-term rate would systematically mis-rank trend-followers against
# high-turnover families -- a bias BETWEEN family types, which is worse
# than a level error.
#
# VALUE DERIVATION: STCG_TAX_RATE above is pinned to a 12% federal
# ordinary-income bracket assumption (Rev. Proc. 2025-32), because
# short-term gains are taxed as ordinary income. Under the same assumption
# the long-term rate is the 0% capital-gains bracket, whose ceiling is set
# close to the top of the 12% ordinary bracket by design. 0.0 is therefore
# the value CONSISTENT with the already-pinned short-term assumption rather
# than an independent choice -- but it means every long-term gain carries no
# modelled tax at all, so it is called out here rather than buried.
# Changing the bracket assumption must change BOTH rates together.
LTCG_TAX_RATE = 0.0

# Calendar-day holding period beyond which a round trip is long-term.
# Strictly greater than: a position held exactly this many days is still
# short-term.
LONG_TERM_HOLDING_DAYS = 365

# Rate credited to idle (uninvested) cash in the ABI-1 account model: if
# actually holding cash, use this small sweep rate; if holding bonds or some
# other risk-free-rate instrument, track friction costs like spread and
# taxes.
#
# WHY THIS MATTERS more than it looks. Crediting the 13-week bill's
# coupon-equivalent yield to idle cash is regime-dependent: the rate has
# ranged from near zero to several percent across the scored history, so the
# size of the effect depends on WHEN a candidate sits out, not only how
# often. A typical brokerage account earns none of the bill rate: uninvested
# balances land in the default sweep, which pays a small fixed rate.
# Crediting the bill rate to an account that is actually holding cash is not
# a level error -- it is a bias BETWEEN strategy types, because it silently
# subsidises exactly the patient, cash-heavy candidates ABI-1 exists to judge
# fairly. Moving the idle credit across this range can be enough to flip the
# sign of a cash-heavy candidate's daily excess Sharpe, so the idle rate is a
# first-order lever on a score.
#
# PROVENANCE: a hand-entered modelling assumption, not an API-sourced quote.
# The difference to a typical real sweep rate is immaterial against the bill
# rate an idle-cash credit would otherwise use (next to it, both round to
# "nothing").
#
# THE OTHER HALF is enforced in account_relative:
# `parked_cash_daily` is the only admitted way to credit a bill/bond/money-
# fund yield instead, and it REQUIRES its friction -- spread and tax on
# interest -- to be declared. There is deliberately no un-frictioned path.
IDLE_SWEEP_APY = 0.001


# --- ABI-1 out-of-sample holdout -------------------------------------------

# Length of the ABI-1 confirmation holdout, in TRADING SESSIONS. Three years
# at 252 sessions/yr. Without this, `account_relative.evaluate` would have no
# out-of-sample gate under any name.
#
# Why three years and not the per-trade construction's HOLDOUT_WINDOW_MONTHS (6):
# the ABI-1 holdout test is a Sharpe test on DAILY sessions, so its power is
# set by the session count. At DSR_PASS_THRESHOLD the annualized daily
# excess Sharpe a holdout must show to be distinguishable from zero is about
# 1.65 over 1 year, 1.17 over 2, 0.95 over 3 and 0.74 over 5 (computed
# against dsr.probabilistic_sharpe_ratio at Normal moments). The in-sample
# bar at DAILY_SR_VARIANCE / ABI1_N_TRIALS is about 1.02 annualized, so
# three years puts the confirmation test slightly EASIER than the search that
# produced the candidate -- the right shape for a confirmation: it must not
# be a second, harder search.
#
# This is a length, not a bar. The bar is DSR_PASS_THRESHOLD above, reused
# unchanged; the holdout introduces no new threshold constant.
ABI1_HOLDOUT_SESSIONS = 756

# --- Options simulator (options-chain simulator families) -----------------------------------

# Hand-curated options-liquidity whitelist -- there is no options-chain
# liquidity data source anywhere in this repo (data_pipeline.py is
# yfinance-equity-only) to derive a floor from the way MIN_POOL_COVERAGE
# was measured, so this is a judgment call, same convention as any other
# batch5/6 curation decision.
OPTIONS_LIQUID_UNDERLYINGS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
]

OPTIONS_UNDERLYING_TIER = {
    "SPY": "index_etf", "QQQ": "index_etf", "IWM": "index_etf", "DIA": "index_etf",
    "AAPL": "mega_cap", "MSFT": "mega_cap", "AMZN": "mega_cap",
    "GOOGL": "mega_cap", "META": "mega_cap", "NVDA": "mega_cap",
}

# Cost per LEG-SIDE (entry, or an active close/roll -- never at a clean
# expiry/assignment, see options_positions.py's own asymmetric cost
# convention), as a fraction of the option's OWN premium (not bps of the
# underlying's price -- option spreads scale with the option's own
# value, unlike an equity's bps-of-price convention). Literature/market-
# convention estimates, NOT measured from a real options-chain quote
# capture (no vendor exists in this pipeline for that) -- mid-points of
# the assumed literature ranges (1-3% index-tier, 2-5% mega-cap-tier).
# A real options-quote capture is the recommended follow-up before trusting
# any passing result at scale, the same discipline the equity quotes need
# before being trusted.
OPTIONS_COST_PCT_OF_PREMIUM = {"index_etf": 0.02, "mega_cap": 0.035}

# Volatility-risk-premium markup on rolling_realized_vol for single-
# stock-tier underlyings (options_vol.implied_vol_for) -- implied
# volatility trades above realized on average, a well-documented
# premium; 1.15 is a conservative starting estimate, not derived from
# this repo's own data (no options-implied-vol vendor exists here).
OPTIONS_IV_REALIZED_VOL_MULTIPLIER = 1.15

# Defensive NAV floor for PMCC specifically (options_positions.
# pmcc_nav_path), as a fraction of the position's own entry_net_debit.
# UNLIKE the cash-secured
# shapes (CSP/covered-call/ratio-spread), which are PROVEN bounded below
# by entry_credit_net via direct no-arbitrage algebra, PMCC's diagonal
# spread has no equivalent proof -- the short leg is margin-covered by
# the long LEAPS, not cash-secured, so there is no K-sized collateral
# term to anchor a bound to. 0.01 (1%) is a small, disclosed backstop,
# not a measured value. A crash-to-near-zero stress test can engage this
# floor (rare, not "never": a non-fresh sub-cycle's long_cost_gross is a
# live re-mark that can legitimately underflow to exactly 0.0 for a
# catastrophically-crashed LEAPS); PMCC_FLOOR_BASIS_EPSILON, above
# pmcc_nav_path's own floor line in options_positions.py, guards the floor
# formula itself.
PMCC_NAV_FLOOR_FRACTION = 0.01

# --- Expanding-window burn-in cap ------------------------------------------

# Maximum years of history, BEFORE the traded instrument's own first
# tradeable date, that may feed an EXPANDING-window statistic (mean, std,
# z-score, ...) computed over an external data series. An expanding window
# seeded from a data series' full history can desensitize completely once it
# accumulates enough pre-inception burn-in: decades of accumulated history
# can leave a z-score that never again crosses its threshold through the
# entire tradeable period, even through an event the mechanism is meant to
# catch.
#
# 10 years (roughly one full economic cycle -- long enough to expose the
# baseline to both an expansion and a contraction, short enough that
# decades of accumulated history cannot neutralize it) is the value
# used: well short of the decades-long burn-in that desensitizes the
# statistic, and chosen on that argument, not for producing the most
# active-looking backtest (picking whichever cap looked best after seeing
# the numbers would be the exact post-hoc tuning this rule exists to
# prevent).
#
# GENERAL RULE: expanding-window candidates must cap pre-inception burn-in.
# The specific value is a judgment call -- flagged per this file's own
# convention (see NEAR_MISS_DSR_MARGIN, OPTIONS_COST_PCT_OF_PREMIUM above)
# rather than silently treated as final.
MAX_PRE_INCEPTION_BURN_IN_YEARS = 10
