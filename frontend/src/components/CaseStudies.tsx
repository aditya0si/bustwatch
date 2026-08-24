import React, { useState, useEffect } from 'react';
import { Bookmark, Calendar, ShieldCheck } from 'lucide-react';
import { BustRecord, fetchHistoricalBusts } from '../services/api';

export const CaseStudies: React.FC = () => {
  const [busts, setBusts] = useState<BustRecord[]>([]);

  useEffect(() => {
    fetchHistoricalBusts().then(setBusts);
  }, []);

  return (
    <div className="bg-[#10172A] border border-slate-800/80 rounded-xl p-5 shadow-lg">
      <div className="flex items-center gap-2 mb-4">
        <Bookmark className="w-5 h-5 text-sky-400" />
        <div>
          <h3 className="text-base font-semibold text-slate-100">Benchmark Historical NWP Forecast Busts</h3>
          <p className="text-xs text-slate-400">
            Validated extreme error regimes where operational medium-range models suffered catastrophic skill loss
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {busts.map((b) => (
          <div
            key={b.event_id}
            className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition-colors"
          >
            <div>
              <div className="flex items-center justify-between text-xs font-mono mb-2">
                <span className="text-sky-400 font-semibold flex items-center gap-1">
                  <Calendar className="w-3.5 h-3.5" />
                  Day {b.lead_time_days} Lead
                </span>
                <span className="text-rose-400 bg-rose-950/40 px-2 py-0.5 rounded border border-rose-800/40 font-bold">
                  {(b.calibrated_bust_prob * 100).toFixed(0)}% Bust Risk
                </span>
              </div>

              <h4 className="text-sm font-bold text-slate-100 mb-1">{b.location_name}</h4>
              <div className="text-[11px] text-amber-400 font-mono mb-2">
                Mechanism: {b.bust_type} | Observed Error: {b.observed_error_z500} m
              </div>

              <p className="text-xs text-slate-300 leading-relaxed line-clamp-4">
                {b.description}
              </p>
            </div>

            <div className="pt-3 mt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-400 font-mono">
              <span>Date: {new Date(b.date).toLocaleDateString()}</span>
              <span className="text-emerald-400 flex items-center gap-1 font-semibold">
                <ShieldCheck className="w-3.5 h-3.5" /> BustWatch Flagged
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
