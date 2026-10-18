"""Shared-calendar two-asset rotation engine for mining rough screens.

This sibling engine keeps multi-asset mechanisms faithful without widening
``SignalResult`` or changing the established single-asset engine's output.
Only the two keys in ``asset_frames`` can be traded or charged costs;
``signal_frames`` are decision inputs only.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd


TargetFn = Callable[[dict[str, pd.DataFrame]], str | None]

_OHLC_COLUMNS = ("Open", "High", "Low", "Close")
_SEGMENT_COLUMNS = (
    "asset", "entry_date", "exit_date", "entry_px", "exit_px", "held",
    "reason", "gross_ret", "net_ret",
)


def _empty_segments() -> pd.DataFrame:
    return pd.DataFrame({
        "asset": pd.Series(dtype="object"),
        "entry_date": pd.Series(dtype="datetime64[ns]"),
        "exit_date": pd.Series(dtype="datetime64[ns]"),
        "entry_px": pd.Series(dtype="float64"),
        "exit_px": pd.Series(dtype="float64"),
        "held": pd.Series(dtype="int64"),
        "reason": pd.Series(dtype="object"),
        "gross_ret": pd.Series(dtype="float64"),
        "net_ret": pd.Series(dtype="float64"),
    })


def _validate_frame(
        key: str, frame: pd.DataFrame,
        expected_calendar: pd.DatetimeIndex | None, *,
        require_positive: bool = True) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"frame {key!r} must be a nonempty OHLC DataFrame")
    missing = [column for column in _OHLC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"frame {key!r} is missing required OHLC columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"frame {key!r} must use a DatetimeIndex calendar")
    if frame.index.tz is not None:
        raise ValueError(
            f"frame {key!r} calendar must be timezone-naive; the mining "
            "pipeline normalizes market calendars before backtesting")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"frame {key!r} calendar must be sorted unique")
    if expected_calendar is not None and not frame.index.equals(expected_calendar):
        raise ValueError(
            f"frame {key!r} must share the identical validated calendar")

    try:
        values = frame.loc[:, _OHLC_COLUMNS].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame {key!r} OHLC values must be numeric and finite") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"frame {key!r} OHLC values must be finite")
    if require_positive:
        open_close = frame.loc[:, ["Open", "Close"]].to_numpy(dtype=float)
        if (open_close <= 0.0).any():
            raise ValueError(f"frame {key!r} Open/Close prices must be positive")
    return frame.index


def _segments_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_segments()
    result = pd.DataFrame(rows, columns=_SEGMENT_COLUMNS)
    result["asset"] = result["asset"].astype("object")
    result["entry_date"] = pd.to_datetime(result["entry_date"])
    result["exit_date"] = pd.to_datetime(result["exit_date"])
    result["entry_px"] = result["entry_px"].astype("float64")
    result["exit_px"] = result["exit_px"].astype("float64")
    result["held"] = result["held"].astype("int64")
    result["reason"] = result["reason"].astype("object")
    result["gross_ret"] = result["gross_ret"].astype("float64")
    result["net_ret"] = result["net_ret"].astype("float64")
    return result


def run_two_asset_rotation(
        asset_frames: Mapping[str, pd.DataFrame], target_fn: TargetFn, *,
        cost_bps_by_asset: Mapping[str, float],
        tax_haircut_rate: float = 0.0,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        ) -> tuple[pd.DataFrame, pd.Series]:
    """Run a next-open rotation between exactly two traded assets.

    On decision bar ``t``, ``target_fn`` receives one dictionary containing
    deep copies of every traded and signal-only frame truncated through
    ``t``. Its asset key or ``None`` executes at ``t+1`` open. A switch closes
    the old asset and opens the new asset at their respective opens on that
    same bar. Unchanged targets do not churn. An active final segment is
    liquidated at the last close.

    Every completed segment is charged that asset's empirical round-trip cost
    once. The optional tax haircut applies only after costs and only to a
    positive net return. Traded and signal-only frames must use the exact same
    sorted, unique, timezone-naive calendar. Signal-only frames can never
    become active or require a cost.

    Every frame's OHLC values must be finite. Traded ``asset_frames`` must
    additionally have strictly positive Open/Close (a traded price can never
    be <= 0). ``signal_frames`` are exempt from that positivity requirement
    -- they are decision inputs only, never priced or traded, and a real
    external series (e.g. a Treasury-bill yield index like ^IRX, which had
    genuine negative closes in March 2020) can legitimately go negative.
    """
    if not isinstance(asset_frames, Mapping) or len(asset_frames) != 2:
        raise ValueError("asset_frames must contain exactly two traded frames")
    if not callable(target_fn):
        raise ValueError("target_fn must be callable")

    assets = dict(asset_frames)
    asset_keys = tuple(assets)
    if any(not isinstance(key, str) or not key for key in asset_keys):
        raise ValueError("asset frame keys must be nonempty strings")

    signals = {} if signal_frames is None else dict(signal_frames)
    overlap = set(assets).intersection(signals)
    if overlap:
        raise ValueError(
            f"signal_frames keys overlap traded assets: {sorted(overlap)}")

    calendar = None
    for key, frame in assets.items():
        calendar = _validate_frame(key, frame, calendar)
    for key, frame in signals.items():
        _validate_frame(key, frame, calendar, require_positive=False)

    if not isinstance(cost_bps_by_asset, Mapping):
        raise ValueError("cost_bps_by_asset must map exactly the traded assets")
    if set(cost_bps_by_asset) != set(asset_keys):
        raise ValueError("cost_bps_by_asset must map exactly the traded assets")
    costs = {}
    for key in asset_keys:
        try:
            cost = float(cost_bps_by_asset[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cost_bps_by_asset values must be finite and nonnegative") from exc
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(
                "cost_bps_by_asset values must be finite and nonnegative")
        costs[key] = cost

    try:
        tax_rate = float(tax_haircut_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("tax_haircut_rate must be finite and in [0, 1)") from exc
    if not math.isfinite(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_haircut_rate must be finite and in [0, 1)")

    decision_inputs = {**assets, **signals}
    active_asset = pd.Series(
        [None] * len(calendar), index=calendar, dtype="object",
        name="active_asset")
    rows = []
    position = None

    def close_position(exit_idx: int, exit_px: float, reason: str) -> None:
        asset = position["asset"]
        gross_ret = exit_px / position["entry_px"] - 1.0
        net_ret = gross_ret - costs[asset] / 1e4
        if net_ret > 0.0 and tax_rate:
            net_ret *= 1.0 - tax_rate
        rows.append({
            "asset": asset,
            "entry_date": calendar[position["entry_idx"]],
            "exit_date": calendar[exit_idx],
            "entry_px": position["entry_px"],
            "exit_px": exit_px,
            "held": exit_idx - position["entry_idx"],
            "reason": reason,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
        })

    for execution_idx in range(1, len(calendar)):
        decision_idx = execution_idx - 1
        truncated = {
            key: frame.iloc[:decision_idx + 1].copy(deep=True)
            for key, frame in decision_inputs.items()
        }
        target = target_fn(truncated)
        if target is not None and target not in assets:
            raise ValueError(
                f"target_fn returned unknown target {target!r}; expected one "
                f"of {list(asset_keys)!r} or None")

        current = None if position is None else position["asset"]
        if target != current:
            if position is not None:
                exit_px = float(
                    assets[current]["Open"].iloc[execution_idx])
                close_position(
                    execution_idx, exit_px,
                    "switch" if target is not None else "to_flat")
                position = None
            if target is not None:
                position = {
                    "asset": target,
                    "entry_idx": execution_idx,
                    "entry_px": float(
                        assets[target]["Open"].iloc[execution_idx]),
                }
        active_asset.iloc[execution_idx] = (
            None if position is None else position["asset"])

    if position is not None:
        final_idx = len(calendar) - 1
        final_px = float(
            assets[position["asset"]]["Close"].iloc[final_idx])
        close_position(final_idx, final_px, "end_of_data")

    return _segments_frame(rows), active_asset
