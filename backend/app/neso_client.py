"""Small client for the NESO Data Portal Daily Demand Update dataset."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://api.neso.energy/api/3/action/datastore_search"
RESOURCE_ID = "177f6fa4-ae49-4182-81ea-0c6b35f26ca6"


class NesoApiError(RuntimeError):
    """Raised when the NESO API cannot return a usable response."""


def fetch_latest_records(limit: int = 10) -> dict:
    """Fetch the newest Daily Demand Update rows from NESO."""
    query = urlencode(
        {
            "resource_id": RESOURCE_ID,
            "limit": limit,
            "sort": "SETTLEMENT_DATE desc,SETTLEMENT_PERIOD desc",
        }
    )
    request = Request(
        f"{API_URL}?{query}",
        headers={"Accept": "application/json", "User-Agent": "GridTrust/1.0"},
    )

    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise NesoApiError(f"NESO API returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        raise NesoApiError(f"Could not connect to the NESO API: {reason}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NesoApiError("NESO API returned an invalid JSON response.") from exc

    if not isinstance(payload, dict):
        raise NesoApiError("NESO API returned an invalid response structure.")
    if not payload.get("success") or not isinstance(payload.get("result"), dict):
        raise NesoApiError("NESO API reported that the data request was unsuccessful.")

    result = payload["result"]
    records = result.get("records")
    if not isinstance(records, list):
        raise NesoApiError("NESO API response did not contain a records list.")

    return {
        "resource_id": result.get("resource_id", RESOURCE_ID),
        "total": result.get("total", len(records)),
        "records": records,
    }
