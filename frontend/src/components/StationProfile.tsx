import React, { useState, useEffect } from 'react';
import { Navigation, TrendingUp } from 'lucide-react';
import { StationMetadata, LeadTimeForecast, fetchStations, fetchLeadProfile } from '../services/api';

interface StationProfileProps {
  selectedCoord: { lat: number; lon: number } | null;
}

export const StationProfile: React.FC<StationProfileProps> = ({ selectedCoord }) => {
  const [stations, setStations] = useState<StationMetadata[]>([]);
  const [selectedStationId, setSelectedStationId] = useState<string>('KORD');
  const [leadProfile, setLeadProfile] = useState<LeadTimeForecast | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchStations().then((data) => {
      setStations(data);
      if (data.length > 0) {
        setSelectedStationId(data[0].station_id);
      }
    });
  }, []);

  useEffect(() => {
    let lat = 41.9742;
    let lon = -87.9073;

    if (selectedCoord) {
      lat = selectedCoord.lat;
      lon = selectedCoord.lon;
      setSelectedStationId('CUSTOM');
    } else {
      const st = stations.find((s) => s.station_id === selectedStationId);
      if (st) {
        lat = st.latitude;
        lon = st.longitude;
      }
    }

    setLoading(true);
    fetchLeadProfile(lat, lon)
      .then((res) => {
        setLeadProfile(res);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load lead profile:', err);
        setLoading(false);
      });
  }, [selectedStationId, selectedCoord, stations]);

  const handleStationChange = (id: string) => {
    setSelectedStationId(id);
  };

  return (
    <div className="bg-[#10172A] border border-slate-800/80 rounded-xl p-5 shadow-lg flex flex-col h-full">
      {/* Header & Station Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <Navigation className="w-5 h-5 text-sky-400" />
            <h3 className="text-base font-semibold text-slate-100">Location Lead-Time Trajectory</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Day 1–10 error amplification & confidence degradation profile
          </p>
        </div>

        {/* Station Dropdown */}
        <div className="flex items-center gap-2">
          <select
            value={selectedStationId}
            onChange={(e) => handleStationChange(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-sky-500 font-medium"
          >
            {selectedStationId === 'CUSTOM' && (
              <option value="CUSTOM">Custom Selected Map Point</option>
            )}
            {stations.map((st) => (
              <option key={st.station_id} value={st.station_id}>
                {st.name} ({st.station_id})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Trajectory Visualizer */}
      <div className="flex-1 min-h-[300px] flex flex-col justify-between">
        {loading ? (
          <div className="h-48 flex items-center justify-center text-slate-400 text-xs">
            <div className="w-6 h-6 border-2 border-sky-500 border-t-transparent rounded-full animate-spin mr-2" />
            Calculating 10-day uncertainty trajectory...
          </div>
        ) : leadProfile ? (
          <div className="space-y-4">
            {/* 10-day cards */}
            <div className="grid grid-cols-2 sm:grid-cols-5 md:grid-cols-10 gap-2">
              {leadProfile.days.map((day, idx) => {
                const conf = leadProfile.confidence_scores[idx];
                const err = leadProfile.expected_errors_z500[idx];
                const risk = leadProfile.risk_levels[idx];

                let borderCol = 'border-emerald-500/30 bg-emerald-950/20';
                let textCol = 'text-emerald-400';
                if (risk === 'MODERATE') {
                  borderCol = 'border-amber-500/30 bg-amber-950/20';
                  textCol = 'text-amber-400';
                } else if (risk === 'ELEVATED') {
                  borderCol = 'border-orange-500/30 bg-orange-950/20';
                  textCol = 'text-orange-400';
                } else if (risk === 'CRITICAL_BUST') {
                  borderCol = 'border-rose-500/30 bg-rose-950/20';
                  textCol = 'text-rose-400';
                }

                return (
                  <div
                    key={day}
                    className={`rounded-lg border p-2.5 flex flex-col items-center justify-between text-center ${borderCol}`}
                  >
                    <span className="text-[11px] font-mono font-bold text-slate-400">D{day}</span>
                    <div className={`text-base font-extrabold my-1 ${textCol}`}>
                      {(conf * 100).toFixed(0)}%
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">{err}m err</span>
                    <span className={`text-[9px] uppercase tracking-tighter mt-1 font-semibold px-1 rounded ${textCol}`}>
                      {risk}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Error Growth Bar Representation */}
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 space-y-2">
              <div className="flex justify-between items-center text-xs text-slate-300">
                <span className="font-semibold flex items-center gap-1.5">
                  <TrendingUp className="w-4 h-4 text-sky-400" />
                  Expected Geopotential Height Error Growth (Z500 Error)
                </span>
                <span className="text-slate-400 font-mono">
                  Coordinates: {leadProfile.latitude.toFixed(2)}°N, {leadProfile.longitude.toFixed(2)}°W
                </span>
              </div>

              {/* Graphical Bar Chart */}
              <div className="h-28 flex items-end gap-2 pt-2 pb-1 border-b border-slate-800">
                {leadProfile.days.map((day, idx) => {
                  const err = leadProfile.expected_errors_z500[idx];
                  const heightPercent = Math.min(100, Math.max(10, (err / 140) * 100));
                  const prob = leadProfile.bust_probabilities[idx];

                  let barColor = 'bg-emerald-500';
                  if (prob > 0.6) barColor = 'bg-rose-500';
                  else if (prob > 0.35) barColor = 'bg-orange-500';
                  else if (prob > 0.15) barColor = 'bg-amber-500';

                  return (
                    <div key={day} className="flex-1 flex flex-col items-center gap-1 h-full justify-end group relative">
                      {/* Tooltip */}
                      <div className="absolute -top-7 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-950 text-sky-300 text-[10px] font-mono px-1.5 py-0.5 rounded border border-sky-500/40 pointer-events-none whitespace-nowrap z-10">
                        {err} m ({(prob * 100).toFixed(0)}% bust)
                      </div>
                      <div
                        style={{ height: `${heightPercent}%` }}
                        className={`w-full rounded-t transition-all duration-300 ${barColor}`}
                      />
                      <span className="text-[10px] font-mono text-slate-400">D{day}</span>
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-between text-[11px] text-slate-500 font-mono">
                <span>0 m (Day 1)</span>
                <span>Lyapunov Divergence Horizon</span>
                <span>~140 m (Day 10)</span>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};
