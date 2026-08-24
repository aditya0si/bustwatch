import React from 'react';
import { Calendar, Zap } from 'lucide-react';

interface LeadTimeSliderProps {
  leadTime: number;
  onChange: (value: number) => void;
}

export const LeadTimeSlider: React.FC<LeadTimeSliderProps> = ({ leadTime, onChange }) => {
  const getRegimeLabel = (d: number) => {
    if (d <= 3) return { label: 'Short-Range (Deterministic Skill High)', color: 'text-emerald-400 bg-emerald-950/60 border-emerald-800/60' };
    if (d <= 7) return { label: 'Medium-Range (Regime Instability Window)', color: 'text-amber-400 bg-amber-950/60 border-amber-800/60' };
    return { label: 'Extended-Range (Nonlinear Chaos Frontier)', color: 'text-rose-400 bg-rose-950/60 border-rose-800/60' };
  };

  const regime = getRegimeLabel(leadTime);

  return (
    <div className="bg-[#10172A] border border-slate-800/80 rounded-xl p-5 shadow-lg backdrop-blur">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-sky-500/10 border border-sky-500/20 rounded-lg text-sky-400">
            <Calendar className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Forecast Lead Time</h3>
            <p className="text-xs text-slate-400">Select target verification horizon (24h to 240h)</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className={`text-xs px-3 py-1 rounded-full border font-medium flex items-center gap-1.5 ${regime.color}`}>
            <Zap className="w-3.5 h-3.5" />
            {regime.label}
          </span>
          <div className="bg-sky-500/20 text-sky-300 font-mono font-bold text-lg px-3.5 py-1 rounded-lg border border-sky-500/40">
            Day {leadTime} <span className="text-xs font-normal text-sky-400/80">({leadTime * 24}h)</span>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <input
          type="range"
          min="1"
          max="10"
          step="1"
          value={leadTime}
          onChange={(e) => onChange(parseInt(e.target.value))}
          className="w-full h-2.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500 focus:outline-none"
        />

        <div className="flex justify-between text-xs font-mono text-slate-400 px-1">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((d) => (
            <button
              key={d}
              onClick={() => onChange(d)}
              className={`transition-colors duration-150 ${
                d === leadTime
                  ? 'text-sky-400 font-bold scale-110'
                  : 'hover:text-slate-200'
              }`}
            >
              D{d}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
