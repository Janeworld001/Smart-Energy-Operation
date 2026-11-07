from __future__ import annotations

import os
import json
import csv
import io
import sqlite3
import statistics
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.neso_client import NesoApiError, fetch_latest_records

DEFAULT_GRID_CONFIG = {
    "candidate_line": 0,
    "risk_thresholds": {"normal_below_percent": 80, "constraint_at_percent": 100},
    "corrective_actions": {"battery": True, "demand_response": True, "redispatch": True},
    "human_approval_required": True,
}

app = FastAPI(title="GridTrust API", version="1.0.0")
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://smart-energy-operation-b2o1.vercel.app",
]
allowed_origins.extend(
    origin.strip().rstrip("/")
    for origin in os.getenv("GRIDTRUST_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
UK_TIME = ZoneInfo("Europe/London")


class GridSnapshot(BaseModel):
    demand_mw: float = Field(28_000, gt=0, description="Current GB national demand in MW")
    wind_mw: float = Field(6_000, ge=0)
    solar_mw: float = Field(1_500, ge=0)
    pumping_mw: float = Field(300, ge=0)
    hour: int = Field(18, ge=0, le=23)
    history: dict[str, float | None] = Field(default_factory=dict)
    history_is_example: bool = True
    observed_at: datetime | None = None
    source_name: str | None = Field(default=None, max_length=120)
    source_fields: list[str] = Field(default_factory=list)
    synthetic_stress_pct: int = Field(0, ge=0, le=200)


class CSVImportRequest(BaseModel):
    csv_text: str = Field(min_length=1, max_length=10_000_000)
    file_name: str = Field("uploaded.csv", max_length=120)


class Decision(BaseModel):
    status: Literal["approved", "modified", "rejected"]
    action: Literal["battery", "demand_response", "redispatch"] | None = None
    note: str = Field("", max_length=500)
    analysis_id: str | None = None
    operator_label: str = Field("Operator", min_length=1, max_length=80)


@dataclass(frozen=True)
class ModelBundle:
    demand_model: object
    uncertainty_model: object
    features: tuple[str, ...]
    conformal_multiplier: float
    metadata: dict
    grid_config: dict


@lru_cache(maxsize=1)
def load_models() -> ModelBundle | None:
    """Load the supplied trained artifacts, or use demo mode if the bundle is absent."""
    project_root = Path(__file__).resolve().parents[2]
    model_dir = Path(os.getenv("GRIDTRUST_MODEL_DIR", project_root / "Models" / "gridtrust_models"))
    demand_path = Path(os.getenv("GRIDTRUST_DEMAND_MODEL", model_dir / "gridtrust_demand_v2.pkl"))
    uncertainty_path = Path(os.getenv("GRIDTRUST_UNCERTAINTY_MODEL", model_dir / "gridtrust_uncertainty.pkl"))
    if not demand_path.exists() and not uncertainty_path.exists():
        return None

    required = {
        "demand model": demand_path,
        "uncertainty model": uncertainty_path,
        "feature list": model_dir / "features_v2.pkl",
        "conformal multiplier": model_dir / "conformal_multiplier.pkl",
        "model metadata": model_dir / "model_metadata.json",
        "grid configuration": model_dir / "grid_config.json",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete GridTrust model bundle: {'; '.join(missing)}")

    try:
        import joblib

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*loading a serialized model.*")
            demand_model = joblib.load(demand_path)
            uncertainty_model = joblib.load(uncertainty_path)
        features = tuple(joblib.load(required["feature list"]))
        multiplier = float(joblib.load(required["conformal multiplier"]))
        metadata = json.loads(required["model metadata"].read_text(encoding="utf-8"))
        grid_config = json.loads(required["grid configuration"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Unable to load GridTrust model bundle: {exc}") from exc

    if not features or not np.isfinite(multiplier) or multiplier <= 0:
        raise RuntimeError("The GridTrust feature list or conformal multiplier is invalid")
    for label, model in (("demand", demand_model), ("uncertainty", uncertainty_model)):
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None and tuple(model_features) != features:
            raise RuntimeError(f"The {label} model feature order does not match features_v2.pkl")
    return ModelBundle(demand_model, uncertainty_model, features, multiplier, metadata, grid_config)


def model_features(snapshot: GridSnapshot, names: tuple[str, ...]) -> pd.DataFrame:
    if snapshot.observed_at is None:
        timestamp = datetime.now(UK_TIME)
    elif snapshot.observed_at.tzinfo is None:
        timestamp = snapshot.observed_at.replace(tzinfo=UK_TIME)
    else:
        timestamp = snapshot.observed_at.astimezone(UK_TIME)
    timestamp = timestamp.replace(hour=snapshot.hour, minute=0, second=0, microsecond=0)
    available: dict[str, float] = {
        "half_hour_slot": float(snapshot.hour * 2),
        "day_of_week": float(timestamp.weekday()),
        "month": float(timestamp.month),
        "day_of_year": float(timestamp.timetuple().tm_yday),
        "is_weekend": float(timestamp.weekday() >= 5),
        "ND": snapshot.demand_mw,
        "EMBEDDED_WIND_GENERATION": snapshot.wind_mw,
        "EMBEDDED_SOLAR_GENERATION": snapshot.solar_mw,
        "PUMP_STORAGE_PUMPING": snapshot.pumping_mw,
        "demand_mw": snapshot.demand_mw,
        "wind_mw": snapshot.wind_mw,
        "solar_mw": snapshot.solar_mw,
        "hour": float(snapshot.hour),
        "hour_sin": float(np.sin(2 * np.pi * snapshot.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * snapshot.hour / 24)),
    }
    available.update({key: value for key, value in snapshot.history.items() if value is not None})
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Missing trained-model inputs: {', '.join(missing)}")
    values = [float(available[name]) for name in names]
    if not np.isfinite(values).all():
        raise ValueError("Trained-model inputs must all be finite numbers")
    return pd.DataFrame([values], columns=names)


def make_forecast(snapshot: GridSnapshot) -> dict:
    bundle = load_models()
    if bundle is None:
        # A transparent demo estimate keeps the application usable without artifacts.
        forecast = snapshot.demand_mw * (
            1.035
            + max(0, 18 - snapshot.hour) * 0.001
            - (snapshot.wind_mw + snapshot.solar_mw) / snapshot.demand_mw * 0.02
        )
        uncertainty = forecast * 0.045
        mode = "demo"
        features = None
    else:
        features = model_features(snapshot, bundle.features)
        forecast = float(np.asarray(bundle.demand_model.predict(features)).reshape(-1)[0])
        if not np.isfinite(forecast) or forecast <= 0:
            raise RuntimeError("Demand model returned an invalid forecast")
        raw_uncertainty = float(np.asarray(bundle.uncertainty_model.predict(features)).reshape(-1)[0])
        uncertainty = abs(raw_uncertainty) * bundle.conformal_multiplier
        if not np.isfinite(uncertainty):
            raise RuntimeError("Uncertainty model returned an invalid estimate")
        mode = "model"

    explanation: list[dict] = []
    explanation_source = "Demo estimates (not SHAP values)"
    if mode == "model":
        import shap

        shap_values = np.asarray(shap.TreeExplainer(bundle.demand_model).shap_values(features)).reshape(-1)
        explanation = [
            {"feature": name.replace("_", " "), "impact_mw": round(float(value), 2)}
            for name, value in zip(bundle.features, shap_values, strict=True)
        ]
        explanation_source = "SHAP TreeExplainer"

    if not explanation:
        explanation = [
            {"feature": "demand", "impact_mw": round((snapshot.demand_mw - 220) * 0.22, 2)},
            {"feature": "wind generation", "impact_mw": round(-snapshot.wind_mw * 0.035, 2)},
            {"feature": "solar generation", "impact_mw": round(-snapshot.solar_mw * 0.04, 2)},
            {"feature": "time of day", "impact_mw": round((snapshot.hour - 12) * 0.55, 2)},
        ]

    result = {
        "horizon": "2 hours",
        "demand_mw": round(forecast, 1),
        "lower_mw": round(max(0, forecast - uncertainty), 1),
        "upper_mw": round(forecast + uncertainty, 1),
        "uncertainty_mw": round(uncertainty, 1),
        "mode": mode,
        "explanation": explanation,
        "explanation_source": explanation_source,
        "history_is_example": snapshot.history_is_example,
    }
    if bundle is not None:
        result.update(
            {
                "model_name": bundle.metadata.get("model_name", "GridTrust demand model"),
                "model_target": bundle.metadata.get("target", "National demand"),
                "conformal_multiplier": round(bundle.conformal_multiplier, 4),
            }
        )
    return result


def simulate_case(
    forecast_mw: float,
    current_demand_mw: float,
    action: str | None,
    grid_config: dict | None = None,
    synthetic_stress_pct: int = 0,
) -> dict:
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="pandapower is required. Install backend/requirements.txt to run grid simulations.",
        ) from exc

    network = pn.case14()
    demand_ratio = forecast_mw / current_demand_mw
    stress_multiplier = 1 + synthetic_stress_pct / 100
    network.load.p_mw *= demand_ratio * stress_multiplier
    network.load.q_mvar *= demand_ratio * stress_multiplier
    for bus, vm_pu in network.ext_grid[["bus", "vm_pu"]].itertuples(index=False, name=None):
        bounds = network.bus.loc[bus]
        network.ext_grid.loc[network.ext_grid.bus == bus, "vm_pu"] = np.clip(
            vm_pu, bounds.min_vm_pu, bounds.max_vm_pu
        )
    for bus, vm_pu in network.gen[["bus", "vm_pu"]].itertuples(index=False, name=None):
        bounds = network.bus.loc[bus]
        network.gen.loc[network.gen.bus == bus, "vm_pu"] = np.clip(
            vm_pu, bounds.min_vm_pu, bounds.max_vm_pu
        )

    try:
        pp.runpp(network, numba=False)
    except pp.LoadflowNotConverged:
        candidate_line = int((grid_config or DEFAULT_GRID_CONFIG).get("candidate_line", 0))
        return {
            "action": action or "baseline",
            "converged": False,
            "max_line_loading_pct": 100.0,
            "candidate_line": candidate_line,
            "candidate_line_loading_pct": 100.0,
            "min_voltage_pu": 0.0,
            "max_voltage_pu": 2.0,
            "constraints": ["Power flow did not converge"],
            "score": 1_000.0,
            "synthetic_stress_pct": synthetic_stress_pct,
        }

    initial_voltage = network.res_bus.vm_pu
    initially_overvoltage = bool((initial_voltage > network.bus.max_vm_pu + 1e-3).any())
    if action == "battery":
        battery_mw = min(24.0, forecast_mw * 0.08)
        if initially_overvoltage:
            pp.create_load(network, bus=8, p_mw=battery_mw, q_mvar=0)
        else:
            pp.create_sgen(network, bus=1, p_mw=battery_mw, q_mvar=0)
    elif action == "demand_response":
        network.load.p_mw *= 0.94
        network.load.q_mvar *= 0.94
    elif action == "redispatch":
        if initially_overvoltage:
            network.gen.vm_pu = network.gen.vm_pu.clip(upper=1.03)
            network.ext_grid.vm_pu = network.ext_grid.vm_pu.clip(upper=1.03)
        else:
            pp.create_sgen(network, bus=5, p_mw=min(20.0, forecast_mw * 0.07), q_mvar=0)

    if action is not None:
        try:
            pp.runpp(network, numba=False)
        except pp.LoadflowNotConverged:
            return {
                "action": action,
                "converged": False,
                "max_line_loading_pct": 100.0,
                "candidate_line": int((grid_config or DEFAULT_GRID_CONFIG).get("candidate_line", 0)),
                "candidate_line_loading_pct": 100.0,
                "min_voltage_pu": 0.0,
                "max_voltage_pu": 2.0,
                "constraints": ["Power flow did not converge"],
                "score": 1_000.0,
                "synthetic_stress_pct": synthetic_stress_pct,
            }

    max_loading = float(network.res_line.loading_percent.max()) if len(network.res_line) else 0
    min_voltage = float(network.res_bus.vm_pu.min())
    max_voltage = float(network.res_bus.vm_pu.max())
    config = grid_config or DEFAULT_GRID_CONFIG
    thresholds = config.get("risk_thresholds", DEFAULT_GRID_CONFIG["risk_thresholds"])
    normal_limit = float(thresholds.get("normal_below_percent", 80))
    constraint_limit = float(thresholds.get("constraint_at_percent", 100))
    candidate_line = int(config.get("candidate_line", 0))
    line_loadings = network.res_line.loading_percent.to_numpy()
    candidate_loading = float(line_loadings[candidate_line]) if 0 <= candidate_line < len(line_loadings) else max_loading
    constraints = []
    if max_loading > normal_limit:
        constraints.append(f"Line loading above normal threshold ({normal_limit:g}%)")
    if max_loading >= constraint_limit:
        constraints.append(f"Line loading at constraint threshold ({constraint_limit:g}%)")
    voltage_tolerance_pu = 1e-3
    voltage_below_limits = np.maximum(
        network.bus.min_vm_pu.to_numpy() - network.res_bus.vm_pu.to_numpy() - voltage_tolerance_pu,
        0,
    )
    voltage_above_limits = np.maximum(
        network.res_bus.vm_pu.to_numpy() - network.bus.max_vm_pu.to_numpy() - voltage_tolerance_pu,
        0,
    )
    if voltage_below_limits.any():
        constraints.append("Voltage below IEEE-14 bus limits")
    if voltage_above_limits.any():
        constraints.append("Voltage above IEEE-14 bus limits")
    score = max(0, max_loading - normal_limit) + float(voltage_below_limits.sum() + voltage_above_limits.sum()) * 100
    return {
        "action": action or "baseline",
        "converged": True,
        "max_line_loading_pct": round(max_loading, 1),
        "candidate_line": candidate_line,
        "candidate_line_loading_pct": round(candidate_loading, 1),
        "min_voltage_pu": round(min_voltage, 3),
        "max_voltage_pu": round(max_voltage, 3),
        "constraints": constraints,
        "score": round(score, 2),
        "synthetic_stress_pct": synthetic_stress_pct,
    }


def database_connection() -> sqlite3.Connection:
    project_root = Path(__file__).resolve().parents[2]
    database_path = Path(os.getenv("GRIDTRUST_DB_PATH", project_root / "data" / "gridtrust.sqlite3"))
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY,
            analysis_id TEXT REFERENCES analyses(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL,
            operator_label TEXT NOT NULL,
            status TEXT NOT NULL,
            action TEXT,
            note TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS decisions_analysis_id_idx ON decisions(analysis_id);
        """
    )
    return connection


def parse_csv_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UK_TIME)
    return timestamp.astimezone(UK_TIME)


def import_demand_csv(csv_text: str, file_name: str) -> dict:
    if len(csv_text.encode("utf-8")) > 10_000_000:
        raise ValueError("CSV file exceeds the 10 MB import limit.")
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV must have a header row.")

    original_headers = reader.fieldnames
    normalized_headers = [
        header.replace("\ufeff", "").strip().lower() if header else ""
        for header in original_headers
    ]
    if len(set(normalized_headers)) != len(normalized_headers):
        raise ValueError("CSV contains duplicate or blank column names.")
    headers = set(normalized_headers)
    settlement_format = {"settlement_date", "settlement_period", "nd"}.issubset(headers)
    if settlement_format:
        source_columns = {
            "demand_mw": "nd",
            "wind_mw": "embedded_wind_generation",
            "solar_mw": "embedded_solar_generation",
            "pumping_mw": "pump_storage_pumping",
        }
        required_headers = {"settlement_date", "settlement_period", "nd"}
        available_source_columns = {"demand_mw"} | {
            output for output, source in source_columns.items() if source in headers
        }
    else:
        source_columns = {
            column: column
            for column in ("demand_mw", "wind_mw", "solar_mw", "pumping_mw")
        }
        required_headers = {"timestamp", "demand_mw"}
        available_source_columns = headers
    missing_headers = required_headers - headers
    if missing_headers:
        raise ValueError(
            "CSV must include timestamp and demand_mw, or NESO settlement_date, "
            "settlement_period, and ND columns. "
            f"Missing: {', '.join(sorted(missing_headers))}."
        )

    records: list[dict] = []
    naive_timestamp_seen = False
    header_lookup = dict(zip(normalized_headers, original_headers, strict=True))
    actual_indicator_column = "forecast_actual_indicator"
    for line_number, row in enumerate(reader, start=2):
        if not any((value or "").strip() for value in row.values() if isinstance(value, str)):
            continue
        if None in row:
            raise ValueError(f"Row {line_number} has more values than the header.")
        values = {key: row.get(original) for key, original in header_lookup.items()}
        if settlement_format:
            if any(values.get(key) is None or not values[key].strip() for key in required_headers):
                raise ValueError(f"Row {line_number} is missing a settlement date, period, or ND value.")
            if (
                actual_indicator_column in values
                and values[actual_indicator_column].strip().upper() != "A"
            ):
                raise ValueError(
                    f"Row {line_number} is not marked as actual data (A). "
                    "Upload historical rows whose FORECAST_ACTUAL_INDICATOR is A."
                )
            try:
                settlement_day = datetime.strptime(
                    values["settlement_date"].strip(), "%Y-%m-%d"
                ).replace(tzinfo=UK_TIME)
                settlement_period = int(values["settlement_period"])
            except ValueError as exc:
                raise ValueError(f"Row {line_number} has an invalid settlement date or period.") from exc
            if not 1 <= settlement_period <= 50:
                raise ValueError(f"Row {line_number} has a settlement period outside the valid range 1–50.")
            timestamp = (
                settlement_day.astimezone(timezone.utc)
                + timedelta(minutes=30 * (settlement_period - 1))
            ).astimezone(UK_TIME)
        else:
            if any(values.get(key) is None or not values[key].strip() for key in required_headers):
                raise ValueError(f"Row {line_number} is missing a timestamp or demand value.")
            raw_timestamp = values["timestamp"].strip()
            try:
                parsed_raw = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
                naive_timestamp_seen = naive_timestamp_seen or parsed_raw.tzinfo is None
                timestamp = parse_csv_timestamp(raw_timestamp)
            except ValueError as exc:
                raise ValueError(f"Row {line_number}: {exc}") from exc

        record = {"timestamp": timestamp}
        for column in ("demand_mw", "wind_mw", "solar_mw", "pumping_mw"):
            source_column = source_columns.get(column)
            if column not in available_source_columns or source_column is None:
                continue
            raw_value = values.get(source_column)
            if raw_value is None or not raw_value.strip():
                raise ValueError(f"Row {line_number} is missing a value in {column}.")
            try:
                numeric_value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Row {line_number} has an invalid number in {column}.") from exc
            if not np.isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(f"Row {line_number} has a negative or non-finite {column}.")
            if column == "demand_mw" and numeric_value == 0:
                raise ValueError(f"Row {line_number} demand_mw must be greater than zero.")
            record[column] = numeric_value
        records.append(record)
        if len(records) > 200_000:
            raise ValueError("CSV exceeds the 200,000-row import limit.")

    if len(records) < 337:
        raise ValueError(
            f"At least 337 half-hour rows are required to calculate the model's "
            f"7-day history features; this file has {len(records)}."
        )
    timestamps = [record["timestamp"].astimezone(timezone.utc) for record in records]
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError("Rows must be in strictly increasing timestamp order with no duplicates.")
    if any(later - earlier != timedelta(minutes=30) for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError("Rows must be consecutive 30-minute readings with no missing intervals.")

    demand = [record["demand_mw"] for record in records]
    rolling_window = demand[-49:-1]
    history = {
        "lag_1": demand[-2],
        "lag_2": demand[-3],
        "lag_48": demand[-49],
        "lag_96": demand[-97],
        "lag_336": demand[-337],
        "rolling_mean_48": statistics.mean(rolling_window),
        "rolling_std_48": statistics.stdev(rolling_window),
    }
    latest = records[-1]
    available_fields = [
        field for field in ("demand_mw", "wind_mw", "solar_mw", "pumping_mw")
        if field in available_source_columns
    ]
    source_name = file_name.replace("\\", "/").rsplit("/", 1)[-1][:120] or "uploaded.csv"
    snapshot = GridSnapshot(
        demand_mw=latest["demand_mw"],
        wind_mw=latest.get("wind_mw", 0),
        solar_mw=latest.get("solar_mw", 0),
        pumping_mw=latest.get("pumping_mw", 0),
        hour=latest["timestamp"].hour,
        history=history,
        history_is_example=False,
        observed_at=latest["timestamp"],
        source_name=source_name,
        source_fields=available_fields,
    )
    warnings_list = []
    if naive_timestamp_seen:
        warnings_list.append("Timestamps without a timezone were interpreted as Europe/London local time.")
    if len(available_fields) < 4:
        missing_optional = sorted({"wind_mw", "solar_mw", "pumping_mw"} - set(available_fields))
        warnings_list.append(
            f"CSV does not include {', '.join(missing_optional)}; keep or enter those current values manually."
        )
    age_minutes = (datetime.now(timezone.utc) - latest["timestamp"].astimezone(timezone.utc)).total_seconds() / 60
    if age_minutes > 60:
        warnings_list.append(f"Latest reading is {round(age_minutes)} minutes old; verify data freshness.")
    elif age_minutes < -5:
        warnings_list.append("Latest reading is in the future; verify the timestamp and timezone.")
    return {
        "snapshot": snapshot.model_dump(mode="json"),
        "rows_imported": len(records),
        "available_fields": available_fields,
        "warnings": warnings_list,
    }


def assess_data_quality(snapshot: GridSnapshot) -> dict:
    warnings_list = []
    if snapshot.history_is_example:
        warnings_list.append("Illustrative example demand history is being used.")
    if snapshot.observed_at is None:
        warnings_list.append("No source timestamp was supplied; data freshness cannot be checked.")
    else:
        observed_at = snapshot.observed_at
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=UK_TIME)
        age_minutes = (datetime.now(timezone.utc) - observed_at.astimezone(timezone.utc)).total_seconds() / 60
        if age_minutes > 60:
            warnings_list.append(f"Latest reading is {round(age_minutes)} minutes old.")
        elif age_minutes < -5:
            warnings_list.append("Latest reading is in the future; verify its timestamp.")
    missing_current_fields = {"wind_mw", "solar_mw", "pumping_mw"} - set(snapshot.source_fields)
    if snapshot.source_name and missing_current_fields:
        warnings_list.append(
            f"CSV did not supply {', '.join(sorted(missing_current_fields))}; current scenario values may be manual."
        )
    previous_demand = snapshot.history.get("lag_1")
    if previous_demand and abs(snapshot.demand_mw - previous_demand) / previous_demand > 0.1:
        warnings_list.append("Current demand differs from the previous half-hour by more than 10%; check for a data gap or unusual change.")
    if snapshot.wind_mw + snapshot.solar_mw > snapshot.demand_mw:
        warnings_list.append("Wind and solar values exceed total demand; verify units and source fields.")
    if snapshot.synthetic_stress_pct:
        warnings_list.append(
            f"Synthetic network stress is enabled (+{snapshot.synthetic_stress_pct}% test-network loading); this does not change the demand forecast."
        )
    return {"status": "review" if warnings_list else "ok", "warnings": warnings_list}


def save_analysis(snapshot: GridSnapshot, result: dict) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    result["analysis_id"] = str(uuid4())
    result["created_at"] = created_at
    connection = database_connection()
    try:
        with connection:
            connection.execute(
                "INSERT INTO analyses (id, created_at, snapshot_json, result_json) VALUES (?, ?, ?, ?)",
                (
                    result["analysis_id"],
                    created_at,
                    json.dumps(snapshot.model_dump(mode="json")),
                    json.dumps(result),
                ),
            )
    finally:
        connection.close()
    return result


def load_history(limit: int = 25) -> list[dict]:
    connection = database_connection()
    try:
        rows = connection.execute(
            "SELECT id, created_at, snapshot_json, result_json FROM analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return []
        analysis_ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in analysis_ids)
        decisions = connection.execute(
            f"SELECT id, analysis_id, created_at, operator_label, status, action, note "
            f"FROM decisions WHERE analysis_id IN ({placeholders}) ORDER BY created_at DESC",
            analysis_ids,
        ).fetchall()
        decisions_by_analysis: dict[str, list[dict]] = {}
        for decision in decisions:
            decisions_by_analysis.setdefault(decision["analysis_id"], []).append(dict(decision))
        history = []
        for row in rows:
            result = json.loads(row["result_json"])
            history.append(
                {
                    "analysis_id": row["id"],
                    "created_at": row["created_at"],
                    "snapshot": json.loads(row["snapshot_json"]),
                    "forecast_mw": result["forecast"]["demand_mw"],
                    "recommendation": result["recommendation"],
                    "data_quality": result["data_quality"],
                    "decisions": decisions_by_analysis.get(row["id"], []),
                }
            )
        return history
    finally:
        connection.close()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/neso-data")
def get_neso_data(limit: int = Query(10, ge=1, le=100)) -> dict:
    """Return the newest rows from NESO's Daily Demand Update dataset."""
    try:
        return fetch_latest_records(limit)
    except NesoApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/import-data")
def import_data(request: CSVImportRequest) -> dict:
    try:
        return import_demand_csv(request.csv_text, request.file_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/analyze")
def analyze(snapshot: GridSnapshot) -> dict:
    try:
        forecast = make_forecast(snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GridTrust forecast failed: {exc}") from exc

    bundle = load_models()
    grid_config = bundle.grid_config if bundle is not None else DEFAULT_GRID_CONFIG
    cases = [
        simulate_case(
            forecast["demand_mw"],
            snapshot.demand_mw,
            action,
            grid_config,
            snapshot.synthetic_stress_pct,
        )
        for action in (None, "battery", "demand_response", "redispatch")
        if action is None or grid_config.get("corrective_actions", {}).get(action, False)
    ]
    baseline = cases[0]
    constrained = bool(baseline["constraints"])
    best = min(cases[1:], key=lambda case: case["score"]) if len(cases) > 1 else baseline
    if not baseline["converged"] and all(not case["converged"] for case in cases[1:]):
        recommendation = {
            "action": None,
            "title": "Simulation inconclusive",
            "reason": "Power flow did not converge for the baseline or tested actions. Do not act on this simulation.",
        }
    elif constrained and best["score"] < baseline["score"] and best is not baseline:
        recommendation = {
            "action": best["action"],
            "title": {
                "battery": "Dispatch battery storage",
                "demand_response": "Activate demand response",
                "redispatch": "Redispatch generation",
            }[best["action"]],
            "reason": f"Lowest simulated constraint score ({best['score']}) across the tested actions.",
        }
    else:
        recommendation = {
            "action": None,
            "title": "Maintain current dispatch",
            "reason": "No actionable constraint detected in the test network.",
        }
    result = {
        "network": "IEEE 14-bus synthetic test network",
        "network_notice": "Synthetic IEEE-14 test network — not the real UK grid.",
        "forecast": forecast,
        "simulation": {"baseline": baseline, "options": cases[1:], "constraint_detected": constrained},
        "recommendation": recommendation,
        "data_quality": assess_data_quality(snapshot),
    }
    return save_analysis(snapshot, result)


@app.post("/api/decisions")
def record_decision(decision: Decision) -> dict:
    entry = {
        "id": str(uuid4()),
        "analysis_id": decision.analysis_id,
        "status": decision.status,
        "action": decision.action,
        "note": decision.note,
        "operator_label": decision.operator_label.strip() or "Operator",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    connection = database_connection()
    try:
        with connection:
            connection.execute(
                "INSERT INTO decisions (id, analysis_id, created_at, operator_label, status, action, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["id"],
                    entry["analysis_id"],
                    entry["created_at"],
                    entry["operator_label"],
                    entry["status"],
                    entry["action"],
                    entry["note"],
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=404, detail="The analysis for this decision was not found.") from exc
    finally:
        connection.close()
    return {**entry, "recorded": True}


@app.get("/api/history")
def list_history() -> dict:
    return {"analyses": load_history()}


@app.get("/api/decisions")
def list_decisions() -> dict:
    connection = database_connection()
    try:
        decisions = connection.execute(
            "SELECT id, analysis_id, created_at, operator_label, status, action, note "
            "FROM decisions ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        return {"decisions": [dict(row) for row in decisions]}
    finally:
        connection.close()
