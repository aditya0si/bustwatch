/**
 * BustWatch API Client Service
 */

export interface StationMetadata {
  station_id: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation_m: number;
  climate_zone: string;
}

export interface LeadTimeForecast {
  latitude: number;
  longitude: number;
  init_time: string;
  days: number[];
  bust_probabilities: number[];
  confidence_scores: number[];
  expected_errors_z500: number[];
  risk_levels: string[];
}

export interface GridRasterResponse {
  lead_time_days: number;
  bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  resolution_deg: number;
  shape: [number, number];
  latitudes: number[];
  longitudes: number[];
  calibrated_bust_probability: number[][];
  raw_bust_probability: number[][];
  confidence_score: number[][];
  expected_error_z500_m: number[][];
  expected_error_t2m_c: number[][];
}

export interface EvalMetrics {
  dataset: {
    n_samples: number;
    overall_bust_rate: number;
    lead_time_range_days: number[];
  };
  metrics_summary: {
    uncalibrated: {
      brier_score: number;
      brier_skill_score: number;
      expected_calibration_error: number;
    };
    calibrated: {
      brier_score: number;
      brier_skill_score: number;
      expected_calibration_error: number;
      roc_auc: number;
      pr_auc: number;
    };
    calibration_improvement: {
      brier_score_reduction_pct: number;
      ece_reduction_pct: number;
    };
  };
  lead_time_breakdown: Record<string, {
    sample_count: number;
    bust_rate: number;
    brier_score: number;
    brier_skill_score: number;
    ece: number;
    roc_auc: number;
  }>;
  feature_importances: Record<string, number>;
  reliability_curve: {
    bin_centers: number[];
    calibrated_observed_freq: number[];
    calibrated_mean_predicted: number[];
    bin_counts: number[];
  };
}

export interface BustRecord {
  event_id: string;
  date: string;
  lead_time_days: number;
  location_name: string;
  latitude: number;
  longitude: number;
  bust_type: string;
  observed_error_z500: number;
  calibrated_bust_prob: number;
  description: string;
}

const API_BASE = '/api/v1';

export async function fetchHealth() {
  const res = await fetch('/health');
  return res.json();
}

export async function fetchStations(): Promise<StationMetadata[]> {
  try {
    const res = await fetch(`${API_BASE}/forecast/stations`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch {
    // Fallback stations
    return [
      { station_id: "KORD", name: "Chicago O'Hare, IL", latitude: 41.9742, longitude: -87.9073, elevation_m: 204.0, climate_zone: "humid_continental" },
      { station_id: "KDEN", name: "Denver International, CO", latitude: 39.8561, longitude: -104.6737, elevation_m: 1656.0, climate_zone: "semi_arid_leeward" },
      { station_id: "KJFK", name: "New York JFK, NY", latitude: 40.6413, longitude: -73.7781, elevation_m: 4.0, climate_zone: "coastal_maritime" },
      { station_id: "KSEA", name: "Seattle Tacoma, WA", latitude: 47.4502, longitude: -122.3088, elevation_m: 132.0, climate_zone: "pacific_northwest" },
      { station_id: "KMCO", name: "Orlando International, FL", latitude: 28.4312, longitude: -81.3081, elevation_m: 29.0, climate_zone: "subtropical" },
      { station_id: "EDDF", name: "Frankfurt Airport, Germany", latitude: 50.0379, longitude: 8.5622, elevation_m: 111.0, climate_zone: "western_europe" },
    ];
  }
}

export async function fetchLeadProfile(lat: number, lon: number): Promise<LeadTimeForecast> {
  const res = await fetch(`${API_BASE}/forecast/lead-profile?latitude=${lat}&longitude=${lon}`);
  if (!res.ok) throw new Error(`Failed to fetch lead profile: ${res.statusText}`);
  return await res.json();
}

export async function fetchConfidenceGrid(leadTime: number): Promise<GridRasterResponse> {
  const res = await fetch(`${API_BASE}/maps/confidence-grid?lead_time_days=${leadTime}&resolution_deg=2.5`);
  if (!res.ok) throw new Error(`Failed to fetch grid: ${res.statusText}`);
  return await res.json();
}

export async function fetchMetrics(): Promise<EvalMetrics> {
  const res = await fetch(`${API_BASE}/evals/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  return await res.json();
}

export async function fetchHistoricalBusts(): Promise<BustRecord[]> {
  const res = await fetch(`${API_BASE}/evals/historical-busts`);
  if (!res.ok) throw new Error(`Failed to fetch historical busts: ${res.statusText}`);
  return await res.json();
}

export async function predictBustRisk(params: {
  lead_time_days: number;
  latitude: number;
  longitude: number;
  ensemble_spread_z500: number;
  ensemble_spread_t2m: number;
  ens_mean_det_diff_z500: number;
  rossby_wave_activity: number;
  blocking_metric: number;
}) {
  const res = await fetch(`${API_BASE}/forecast/bust-risk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
  return await res.json();
}
