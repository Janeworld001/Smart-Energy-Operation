# GridTrust

GridTrust is a local decision-support prototype for exploring short-term electricity-demand scenarios. It combines a two-hour Great Britain demand forecast, an explanation of the forecast, and a power-flow simulation on the IEEE 14-bus test network. An operator can compare simulated actions and record whether they accept, change, or reject a suggestion.

The project is intended for demonstration and development, not for controlling a power system. The dashboard starts with illustrative values; review and replace them before interpreting an analysis.

## What it does

- Forecasts national demand two hours ahead at the model's 30-minute data resolution.
- Shows a forecast range and feature contributions. With the supplied model bundle, explanations are calculated with SHAP; without it, the app labels both the forecast and its illustrative explanation as demo output.
- Runs a baseline and corrective-action scenarios on a synthetic IEEE 14-bus network.
- Checks imported or manually entered data for several common quality issues and displays warnings for operator review.
- Saves analysis inputs and outputs, plus operator decisions, in a local SQLite database.

GridTrust does not connect to a live grid or automatically dispatch generation, batteries, or demand response.

## Project layout

```text
GridTrust/
├── Models/
│   └── gridtrust_models/       # Trained models and their configuration
├── backend/
│   ├── app/main.py             # FastAPI API, forecasting, simulation, and persistence
│   ├── requirements.txt
│   └── tests/test_main.py
├── data/
│   └── gridtrust.sqlite3       # Local analysis and decision history
└── frontend/
    ├── src/App.tsx             # Dashboard
    ├── src/api.ts              # Frontend API client and data types
    └── package.json
```

## Requirements

- Windows PowerShell (commands below use PowerShell syntax)
- Python with `venv` support
- Node.js and npm

The backend dependencies include FastAPI, Uvicorn, pandas, NumPy, pandapower, SHAP, and joblib. XGBoost is pinned to `3.1.2` to match the supplied serialized model artifacts. The frontend is a React and TypeScript app served by Vite.

## Run locally

From the project root, create a virtual environment and install the backend packages:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

Start the API from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir .\backend --reload
```

The API listens at `http://127.0.0.1:8000`. Its health check is at `http://127.0.0.1:8000/health`, and FastAPI's interactive API documentation is at `http://127.0.0.1:8000/docs`.

In a second terminal, install and start the frontend:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend uses `http://127.0.0.1:8000` as its default API address. To point it elsewhere, set `VITE_API_URL` before starting Vite:

```powershell
$env:VITE_API_URL = "http://127.0.0.1:8000"
npm run dev
```

The development API allows browser requests from `localhost:5173` and `127.0.0.1:5173`. If you change the frontend's origin, update the backend CORS configuration in `backend/app/main.py` as well.

## Forecast model and artifacts

By default, the backend looks for the model bundle in `Models/gridtrust_models/`:

| File | Purpose |
| --- | --- |
| `gridtrust_demand_v2.pkl` | Trained demand forecast model |
| `gridtrust_uncertainty.pkl` | Model used to estimate forecast uncertainty |
| `features_v2.pkl` | Ordered model feature names |
| `conformal_multiplier.pkl` | Calibration multiplier for the forecast range |
| `model_metadata.json` | Model description and training metadata |
| `grid_config.json` | Simulation thresholds and enabled corrective actions |

The default metadata describes a two-hour XGBoost forecast for Great Britain national demand, trained using 30-minute data. The uncertainty estimate is multiplied by the supplied conformal calibration value to produce the displayed lower and upper bounds. The app loads these artifacts; it does not train or modify them.

If neither forecast-model file is present, GridTrust uses a clearly labeled illustrative demo estimate so that the interface can still be explored. If one model file is present but the bundle is incomplete, or an artifact cannot be loaded, the backend reports an error rather than silently switching modes. Model explanation in trained mode uses SHAP's `TreeExplainer`.

Only load model artifacts from a source you trust. Pickle/joblib files are serialized Python objects and should not be treated as safe input from an unknown source.

### Pointing to another model bundle

Set these environment variables before starting the backend:

| Variable | Default | Description |
| --- | --- | --- |
| `GRIDTRUST_MODEL_DIR` | `Models/gridtrust_models` | Directory containing the feature list, calibration value, metadata, and grid configuration |
| `GRIDTRUST_DEMAND_MODEL` | `gridtrust_demand_v2.pkl` inside the model directory | Demand model file path |
| `GRIDTRUST_UNCERTAINTY_MODEL` | `gridtrust_uncertainty.pkl` inside the model directory | Uncertainty model file path |

The bundle still needs `features_v2.pkl`, `conformal_multiplier.pkl`, `model_metadata.json`, and `grid_config.json` in `GRIDTRUST_MODEL_DIR`.

## Entering scenario data

The dashboard offers editable fields for current demand, wind generation, solar generation, pumped-storage pumping, and hour of day. Its initial readings and history are examples, not current system measurements. For the trained model, supply the seven history values below as well:

| Field | Meaning |
| --- | --- |
| `lag_1` | Demand one 30-minute interval ago |
| `lag_2` | Demand two 30-minute intervals ago |
| `lag_48` | Demand 24 hours ago |
| `lag_96` | Demand 48 hours ago |
| `lag_336` | Demand seven days ago |
| `rolling_mean_48` | Mean demand over the previous 48 intervals |
| `rolling_std_48` | Sample standard deviation over those 48 intervals |

The UI requires these values to be present, finite, and non-negative before enabling analysis. If a required feature is missing from the model input, the API returns a validation error. Enter scenario values in MW. The current model does not use temperature or grid imports as adjustable forecast inputs.

### Import a CSV

The dashboard's Grid simulation section accepts a CSV file. The import endpoint expects these column names:

```csv
timestamp,demand_mw,wind_mw,solar_mw,pumping_mw
2025-01-01T00:00:00+00:00,28000,6000,1500,300
```

`timestamp` and `demand_mw` are required. The other three columns are optional; if omitted, the dashboard keeps its manually entered values for those current-generation inputs. Column names are trimmed and case-insensitive. If an optional column is included, each non-empty data row must have a valid value for it.

Import requirements and behavior:

- At least **337 consecutive readings**, one per half hour, are needed to calculate the seven-day lag.
- Timestamps must be in strictly increasing order without duplicates or missing half-hour intervals. ISO-8601 timestamps are accepted.
- Timestamps with an explicit timezone are converted to Europe/London time. Timestamps without a timezone are interpreted as Europe/London local time and flagged in the import warnings.
- Demand must be greater than zero; other numeric values must be finite and non-negative.
- The import is limited to 10 MB and 200,000 rows.
- The latest row supplies the current reading. Demand history features are calculated from the preceding readings.
- The app checks data freshness and reports missing optional fields or other quality concerns. Warnings are prompts for review, not proof that the data is correct.

After import, check the reported warnings, confirm the current values, and run a new analysis to use the imported scenario.

## Understanding the simulation

The backend builds a pandapower IEEE 14-bus test case and scales its loads using the ratio of forecast demand to current demand. The **Synthetic test-network stress** control adds a further load multiplier to that network only. It does not change the demand forecast.

GridTrust compares a baseline with the enabled battery, demand-response, and redispatch cases. The default configuration flags line loadings above 80% as beyond the normal threshold and loadings at or above 100% as a constraint. It also checks voltage against the test network's bus limits. When a baseline constraint is detected, the app can suggest the tested action with the lowest simulated constraint score.

These are simplified scenarios on a modified IEEE-14 network, not predictions of real transmission conditions. The network's MW values and action results must not be interpreted as measurements or instructions for the GB grid. A recommendation is advisory; the interface allows an operator to approve it, modify the action, or reject it, and records that choice.

## API reference

The backend exposes these endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Basic service health check |
| `POST` | `/api/import-data` | Validate CSV text and return a scenario snapshot and import warnings |
| `POST` | `/api/analyze` | Forecast demand, run the network scenarios, and save the analysis |
| `POST` | `/api/decisions` | Save an operator decision associated with an analysis |
| `GET` | `/api/history` | Return the 25 most recent analyses and their decisions |
| `GET` | `/api/decisions` | Return the 50 most recent decisions |

Request and response schemas are available at `/docs` while the backend is running. For CSV import, send JSON with `csv_text` and `file_name`. For analysis, send a snapshot containing the current readings, history, timestamp/source information where available, and optional `synthetic_stress_pct`. Decisions use a status of `approved`, `modified`, or `rejected`; an action can be `battery`, `demand_response`, `redispatch`, or `null`.

## Local storage and configuration

Analyses and decisions are stored in SQLite at `data/gridtrust.sqlite3` by default. The database and its tables are created when the backend first connects. An analysis record includes its scenario snapshot and result; a decision records the operator label, status, selected action, note, and associated analysis ID.

To use a different database file, set `GRIDTRUST_DB_PATH` before starting the backend. For example:

```powershell
$env:GRIDTRUST_DB_PATH = "C:\GridTrustData\gridtrust.sqlite3"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir .\backend --reload
```

This is local application storage, not a tamper-resistant or production audit system. Back up or remove the database using your normal data-handling process; it may contain scenario inputs and operator-entered notes.

## Tests and frontend build

Run the backend regression tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\backend\tests -v
```

The tests cover CSV history-feature calculation and validation, synthetic stress behavior, and persistence of analyses and decisions. They use a temporary database.

To type-check and build the frontend:

```powershell
Set-Location .\frontend
npm run build
```

## Limitations and safe use

- The supplied initial inputs are illustrative. No live NESO or other operational data feed is connected.
- The IEEE 14-bus network is a synthetic test network and does not represent the physical GB transmission system.
- Data-quality checks identify selected issues but cannot establish the accuracy, provenance, or completeness of a data source.
- Forecast accuracy and uncertainty estimates depend on the supplied model bundle and its training data; this README makes no operational performance guarantee.
- Operator labels are self-reported. The local database is not an authenticated, tamper-proof audit trail.
- The app has no user authentication or access control. Do not expose it as a production service or use it to control equipment.

Before any real-world use, the forecasting and network models, data pipeline, operator identity and access controls, and safety procedures would all need independent validation and approval.
