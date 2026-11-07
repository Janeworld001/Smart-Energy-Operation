import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import neso_client


class NesoClientTests(unittest.TestCase):
    def test_fetch_latest_records_requests_sorted_daily_demand_rows(self):
        payload = {
            "success": True,
            "result": {
                "resource_id": neso_client.RESOURCE_ID,
                "total": 42,
                "records": [{"SETTLEMENT_DATE": "2026-11-06", "ND": 28000}],
            },
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch.object(neso_client, "urlopen", return_value=FakeResponse()) as open_url:
            result = neso_client.fetch_latest_records(5)

        request = open_url.call_args.args[0]
        query = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(query["resource_id"], [neso_client.RESOURCE_ID])
        self.assertEqual(query["limit"], ["5"])
        self.assertEqual(
            query["sort"], ["SETTLEMENT_DATE desc,SETTLEMENT_PERIOD desc"]
        )
        self.assertEqual(result["records"], payload["result"]["records"])
        self.assertEqual(result["total"], 42)

    def test_fetch_latest_records_rejects_unsuccessful_api_response(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return b'{"success": false}'

        with patch.object(neso_client, "urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(neso_client.NesoApiError, "unsuccessful"):
                neso_client.fetch_latest_records()


if __name__ == "__main__":
    unittest.main()
