import React, { useState } from 'react';
import { Sliders, ShieldAlert, Play } from 'lucide-react';
import { predictBustRisk } from '../services/api';

export const RegimeSimulator: React.FC = () => {
  const [leadTime, setLeadTime] = useState<number>(6);
  const [spreadZ500, setSpreadZ500] = useState<number>(65);
  const [spreadT2M, setSpreadT2M] = useState<number>(3.5);
  const [ensMeanDiff, setEnsMeanDiff] = useState<number>(24);
  const [rossbyActivity, setRossbyActivity] = useState<number>(18.0);
  const [blockingMetric, setBlockingMetric] = useState<number>(1.8);

  const [prediction, setPrediction] = useState<{
    prob: number;
    conf: number;
    errZ500: number;
    errT2M: number;
    risk: string;
  } | null>({
    prob: 0.68,
    conf: 0.32,
    errZ500: 92.4,
    errT2M: 4.8,
    risk: 'CRITICAL_BUST',
  });

  const [loading, setLoading] = useState<boolean>(false);

  const handleSimulate = async () => {
    setLoading(true);
    try {
      const res = await predictBustRisk({
        lead_time_days: leadTime,
        latitude: 45.0,
        longitude: -90.0,
        ensemble_spread_z500: spreadZ500,
        ensemble_spread_t2m: spreadT2M,
        ens_mean_det_diff_z500: ensMeanDiff,
        rossby_wave_activity: rossbyActivity,
        blocking_metric: blockingMetric,
      });

      setPrediction({
        prob: res.bust_probability_calibrated,
        conf: res.confidence_score,
        errZ500: res.expected_error_z500_m,
        errT2M: res.expected_error_t2m_c,
        risk: res.risk_level,
      });
    } catch (err) {
      console.error('Simulation error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#10172A] border border-slate-800/80 rounded-xl p-5 shadow-lg flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-sky-400" />
            <h3 className="text-base font-semibold text-slate-100">Atmospheric Regime Stress Simulator</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Test how Rossby wave breaking, blocking ridges, and spread-skill divergence trigger forecast busts
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Controls Column */}
        <div className="space-y-3.5 bg-slate-900/60 p-4 rounded-xl border border-slate-800">
          {/* Lead Time Slider */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-300">Lead Time Horizon</span>
              <span className="text-sky-400 font-mono font-bold">Day {leadTime}</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              value={leadTime}
              onChange={(e) => setLeadTime(parseInt(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
          </div>

          {/* Spread Z500 */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-300">Ensemble Spread Z500 (m)</span>
              <span className="text-sky-400 font-mono font-bold">{spreadZ500} m</span>
            </div>
            <input
              type="range"
              min="10"
              max="120"
              value={spreadZ500}
              onChange={(e) => setSpreadZ500(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
          </div>

          {/* Spread T2M */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-300">Ensemble Spread 2m Temp (°C)</span>
              <span className="text-sky-400 font-mono font-bold">{spreadT2M.toFixed(1)} °C</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="8.0"
              step="0.1"
              value={spreadT2M}
              onChange={(e) => setSpreadT2M(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
          </div>

          {/* Rossby Wave Activity */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-300">Rossby Wave Packet Activity</span>
              <span className="text-sky-400 font-mono font-bold">{rossbyActivity} m2/s2</span>
            </div>
            <input
              type="range"
              min="0"
              max="35"
              step="0.5"
              value={rossbyActivity}
              onChange={(e) => setRossbyActivity(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
          </div>

          {/* Tibaldi-Molteni Blocking */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-300">Tibaldi-Molteni Blocking Index</span>
              <span className="text-amber-400 font-mono font-bold">+{blockingMetric.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="-1.5"
              max="3.5"
              step="0.1"
              value={blockingMetric}
              onChange={(e) => setBlockingMetric(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* Spread-Skill Divergence */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-300">Spread-Skill Divergence</span>
              <span className="text-sky-400 font-mono font-bold">{ensMeanDiff} m</span>
            </div>
            <input
              type="range"
              min="0"
              max="60"
              value={ensMeanDiff}
              onChange={(e) => setEnsMeanDiff(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
          </div>

          <button
            onClick={handleSimulate}
            disabled={loading}
            className="w-full mt-2 py-2 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded-lg text-xs flex items-center justify-center gap-2 transition-all shadow-md active:scale-98 disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {loading ? 'Evaluating Model...' : 'Simulate Atmospheric State'}
          </button>
        </div>

        {/* Prediction Results Column */}
        <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div className="space-y-3">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
              Inferred Forecast Bust Vulnerability
            </span>

            {prediction && (
              <>
                <div className="flex items-center justify-between p-3 bg-slate-950/80 rounded-lg border border-slate-800">
                  <div>
                    <span className="text-[11px] text-slate-400">Calibrated Bust Risk</span>
                    <div className="text-2xl font-black font-mono text-rose-400 mt-0.5">
                      {(prediction.prob * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-[11px] text-slate-400">Operational Confidence</span>
                    <div className="text-2xl font-black font-mono text-emerald-400 mt-0.5">
                      {(prediction.conf * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Expected Z500 Error</span>
                    <span className="text-sm font-bold font-mono text-sky-400">{prediction.errZ500.toFixed(1)} m</span>
                  </div>
                  <div className="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Expected T2M Error</span>
                    <span className="text-sm font-bold font-mono text-amber-400">±{prediction.errT2M.toFixed(1)} °C</span>
                  </div>
                </div>

                <div className="p-2.5 rounded-lg border text-xs flex items-center gap-2 bg-rose-950/20 border-rose-800/40 text-rose-300 font-medium">
                  <ShieldAlert className="w-4 h-4 text-rose-400 flex-shrink-0" />
                  <span>
                    <strong>{prediction.risk}</strong>: Severe regime shift threshold exceeded. Climatological verification likely to diverge.
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="text-[11px] text-slate-500 pt-2 border-t border-slate-800/60">
            Model: LightGBM GBDT with Isotonic Calibration (Lyapunov Divergence Horizon: 0.25/day)
          </div>
        </div>
      </div>
    </div>
  );
};
