import csv
import io
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main


class GridTrustWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_environment = patch.dict(
            os.environ,
            {"GRIDTRUST_DB_PATH": str(Path(self.temp_dir.name) / "gridtrust.sqlite3")},
        )
        self.database_environment.start()

    def tearDown(self):
        self.database_environment.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def make_csv(row_count=337, missing_row=None):
        london = ZoneInfo("Europe/London")
        latest = datetime.now(london).replace(minute=0 if datetime.now(london).minute < 30 else 30, second=0, microsecond=0)
        latest -= timedelta(minutes=30)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "demand_mw", "wind_mw", "solar_mw", "pumping_mw"])
        for index in range(row_count):
            if index == missing_row:
                continue
            timestamp = latest - timedelta(minutes=30 * (row_count - 1 - index))
            writer.writerow([timestamp.isoformat(), 28000 + index, 6000, 1500, 300])
        return output.getvalue()

    def test_csv_import_builds_lags_and_rolling_features(self):
        result = main.import_demand_csv(self.make_csv(), "historical.csv")
        snapshot = result["snapshot"]

        self.assertEqual(result["rows_imported"], 337)
        self.assertEqual(snapshot["history"]["lag_1"], 28335)
        self.assertEqual(snapshot["history"]["lag_336"], 28000)
        self.assertAlmostEqual(snapshot["history"]["rolling_mean_48"], 28311.5)
        self.assertEqual(snapshot["source_fields"], ["demand_mw", "wind_mw", "solar_mw", "pumping_mw"])
        self.assertFalse(snapshot["history_is_example"])

    def test_csv_import_rejects_missing_intervals_and_short_history(self):
        with self.assertRaisesRegex(ValueError, "consecutive 30-minute"):
            main.import_demand_csv(self.make_csv(row_count=338, missing_row=100), "gap.csv")
        with self.assertRaisesRegex(ValueError, "At least 337"):
            main.import_demand_csv(self.make_csv(row_count=336), "short.csv")

    def test_analysis_stress_and_decisions_are_persisted(self):
        forecast = {"demand_mw": 28000}
        snapshot = main.GridSnapshot(
            demand_mw=28000,
            wind_mw=6000,
            solar_mw=1500,
            pumping_mw=300,
            hour=18,
            history={
                "lag_1": 27900,
                "lag_2": 27850,
                "lag_48": 27500,
                "lag_96": 28000,
                "lag_336": 27000,
                "rolling_mean_48": 27800,
                "rolling_std_48": 850,
            },
        )
        with patch.object(main, "make_forecast", return_value=forecast), patch.object(main, "load_models", return_value=None):
            normal = main.analyze(snapshot)
            stressed = main.analyze(snapshot.model_copy(update={"synthetic_stress_pct": 200}))

        self.assertFalse(normal["simulation"]["constraint_detected"])
        self.assertTrue(stressed["simulation"]["constraint_detected"])
        self.assertEqual(stressed["recommendation"]["action"], "demand_response")
        self.assertEqual(len(main.list_history()["analyses"]), 2)

        decision = main.record_decision(
            main.Decision(
                status="approved",
                action="demand_response",
                analysis_id=stressed["analysis_id"],
                operator_label="Test operator",
            )
        )
        self.assertTrue(decision["recorded"])
        history = main.list_history()["analyses"]
        self.assertEqual(history[0]["decisions"][0]["operator_label"], "Test operator")
        self.assertEqual(history[0]["decisions"][0]["status"], "approved")


if __name__ == "__main__":
    unittest.main()
