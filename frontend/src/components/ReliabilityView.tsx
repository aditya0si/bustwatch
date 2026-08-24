import React, { useState, useEffect } from 'react';
import { Target, CheckCircle2, Award, ArrowDownRight, BarChart2 } from 'lucide-react';
import { EvalMetrics, fetchMetrics } from '../services/api';

export const ReliabilityView: React.FC = () => {
  const [metrics, setMetrics] = useState<EvalMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchMetrics()
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching metrics:', err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="bg-[#10172A] border border-slate-800/80 rounded-xl p-5 shadow-lg flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-sky-400" />
            <h3 className="text-base font-semibold text-slate-100">Model Verification & Calibration</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Brier Skill Score (BSS), Expected Calibration Error (ECE), and Reliability Curve
          </p>
        </div>
      </div>

      {loading ? (
        <div className="h-48 flex items-center justify-center text-slate-400 text-xs">
          <div className="w-6 h-6 border-2 border-sky-500 border-t-transparent rounded-full animate-spin mr-2" />
          Loading verification metrics...
        </div>
      ) : metrics ? (
        <div className="space-y-4">
          {/* Key Metric Scorecards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
              <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                Brier Score (BS)
              </span>
              <div className="text-xl font-bold text-sky-400 font-mono mt-0.5">
                {metrics.metrics_summary.calibrated.brier_score.toFixed(4)}
              </div>
              <span className="text-[10px] text-emerald-400 flex items-center gap-0.5 mt-1 font-mono">
                <CheckCircle2 className="w-3 h-3" /> Lower is better (ideal: 0)
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
              <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                Brier Skill Score (BSS)
              </span>
              <div className="text-xl font-bold text-emerald-400 font-mono mt-0.5">
                +{metrics.metrics_summary.calibrated.brier_skill_score.toFixed(3)}
              </div>
              <span className="text-[10px] text-emerald-400 flex items-center gap-0.5 mt-1 font-mono">
                <Award className="w-3 h-3" /> Over Climatology Reference
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
              <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                Expected Calib. Error (ECE)
              </span>
              <div className="text-xl font-bold text-amber-400 font-mono mt-0.5">
                {metrics.metrics_summary.calibrated.expected_calibration_error.toFixed(4)}
              </div>
              <span className="text-[10px] text-sky-400 flex items-center gap-0.5 mt-1 font-mono">
                <ArrowDownRight className="w-3 h-3" /> {metrics.metrics_summary.calibration_improvement.ece_reduction_pct}% calibrated drop
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
              <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                ROC-AUC Score
              </span>
              <div className="text-xl font-bold text-indigo-400 font-mono mt-0.5">
                {metrics.metrics_summary.calibrated.roc_auc.toFixed(3)}
              </div>
              <span className="text-[10px] text-indigo-400 flex items-center gap-0.5 mt-1 font-mono">
                PR-AUC: {metrics.metrics_summary.calibrated.pr_auc.toFixed(3)}
              </span>
            </div>
          </div>

          {/* Interactive Reliability Diagram Curve */}
          <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 space-y-2">
            <div className="flex justify-between items-center text-xs text-slate-300">
              <span className="font-semibold flex items-center gap-1.5">
                <BarChart2 className="w-4 h-4 text-sky-400" />
                Reliability Diagram ($P_{'{predicted}'}$ vs. Observed Event Frequency)
              </span>
              <span className="text-slate-400 text-[11px]">
                Evaluated on {metrics.dataset.n_samples} verification points
              </span>
            </div>

            {/* SVG Reliability Plot */}
            <div className="relative h-44 w-full bg-slate-950/80 rounded-lg border border-slate-800/80 p-3">
              <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
                {/* Grid Lines */}
                <line x1="0" y1="0" x2="100" y2="0" stroke="#1e293b" strokeWidth="0.5" />
                <line x1="0" y1="25" x2="100" y2="25" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="1,1" />
                <line x1="0" y1="50" x2="100" y2="50" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="1,1" />
                <line x1="0" y1="75" x2="100" y2="75" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="1,1" />
                <line x1="0" y1="100" x2="100" y2="100" stroke="#334155" strokeWidth="0.8" />

                <line x1="0" y1="0" x2="0" y2="100" stroke="#334155" strokeWidth="0.8" />
                <line x1="25" y1="0" x2="25" y2="100" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="1,1" />
                <line x1="50" y1="0" x2="50" y2="100" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="1,1" />
                <line x1="75" y1="0" x2="75" y2="100" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="1,1" />
                <line x1="100" y1="0" x2="100" y2="100" stroke="#1e293b" strokeWidth="0.5" />

                {/* Perfect Diagonal Line (y = x) */}
                <line x1="0" y1="100" x2="100" y2="0" stroke="#64748b" strokeWidth="1.2" strokeDasharray="2,2" />

                {/* Calibrated Polyline */}
                {metrics.reliability_curve && metrics.reliability_curve.bin_centers.length > 0 && (
                  <>
                    <polyline
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2"
                      points={metrics.reliability_curve.bin_centers
                        .map((x, i) => {
                          const y = metrics.reliability_curve.calibrated_observed_freq[i] ?? x;
                          return `${x * 100},${100 - y * 100}`;
                        })
                        .join(' ')}
                    />
                    {metrics.reliability_curve.bin_centers.map((x, i) => {
                      const y = metrics.reliability_curve.calibrated_observed_freq[i] ?? x;
                      return (
                        <circle
                          key={i}
                          cx={x * 100}
                          cy={100 - y * 100}
                          r="2.5"
                          fill="#10b981"
                          stroke="#064e3b"
                          strokeWidth="0.5"
                        />
                      );
                    })}
                  </>
                )}
              </svg>
            </div>

            <div className="flex justify-between items-center text-[11px] font-mono text-slate-400 pt-1">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-slate-400 border-dashed border-t border-slate-400 inline-block" />
                Perfect Reliability (y = x)
              </span>
              <span className="flex items-center gap-1.5 text-emerald-400">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
                Calibrated GBDT Probability (Isotonic)
              </span>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
