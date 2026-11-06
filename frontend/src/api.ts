export type Snapshot = {
  demand_mw: number;
  wind_mw: number;
  solar_mw: number;
  pumping_mw: number;
  hour: number;
  history: Record<string, number | null>;
  history_is_example: boolean;
  observed_at: string | null;
  source_name?: string | null;
  source_fields: string[];
  synthetic_stress_pct: number;
};

export type Forecast = {
  horizon: string;
  demand_mw: number;
  lower_mw: number;
  upper_mw: number;
  uncertainty_mw: number;
  mode: "demo" | "model";
  model_name?: string;
  model_target?: string;
  conformal_multiplier?: number;
  history_is_example: boolean;
  explanation: { feature: string; impact_mw: number }[];
  explanation_source: string;
};

export type SimulationCase = {
  action: string;
  converged: boolean;
  max_line_loading_pct: number;
  min_voltage_pu: number;
  max_voltage_pu: number;
  constraints: string[];
  score: number;
  synthetic_stress_pct: number;
};

export type Analysis = {
  analysis_id: string;
  created_at: string;
  network: string;
  network_notice: string;
  forecast: Forecast;
  simulation: {
    baseline: SimulationCase;
    options: SimulationCase[];
    constraint_detected: boolean;
  };
  recommendation: { action: string | null; title: string; reason: string };
  data_quality: { status: "ok" | "review"; warnings: string[] };
};

export type CSVImportResult = {
  snapshot: Snapshot;
  rows_imported: number;
  available_fields: string[];
  warnings: string[];
};

export type HistoryItem = {
  analysis_id: string;
  created_at: string;
  snapshot: Snapshot;
  forecast_mw: number;
  recommendation: Analysis["recommendation"];
  data_quality: Analysis["data_quality"];
  decisions: {
    id: string;
    created_at: string;
    operator_label: string;
    status: "approved" | "modified" | "rejected";
    action: string | null;
    note: string;
  }[];
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const analyze = (snapshot: Snapshot) =>
  request<Analysis>("/api/analyze", {
    method: "POST",
    body: JSON.stringify(snapshot),
  });

export const importCsv = (csvText: string, fileName: string) =>
  request<CSVImportResult>("/api/import-data", {
    method: "POST",
    body: JSON.stringify({ csv_text: csvText, file_name: fileName }),
  });

export const submitDecision = (decision: {
  status: "approved" | "modified" | "rejected";
  action: string | null;
  analysis_id: string;
  operator_label: string;
  note?: string;
}) =>
  request<{ recorded: boolean }>("/api/decisions", {
    method: "POST",
    body: JSON.stringify(decision),
  });

export const getHistory = () => request<{ analyses: HistoryItem[] }>("/api/history");
