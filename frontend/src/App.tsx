import { useState } from 'react';
import {
  CloudLightning,
  Layers,
  BarChart3,
  Sliders,
  Bookmark,
} from 'lucide-react';
import { LeadTimeSlider } from './components/LeadTimeSlider';
import { SpatialMap } from './components/SpatialMap';
import { StationProfile } from './components/StationProfile';
import { ReliabilityView } from './components/ReliabilityView';
import { RegimeSimulator } from './components/RegimeSimulator';
import { CaseStudies } from './components/CaseStudies';

export function App() {
  const [leadTime, setLeadTime] = useState<number>(5);
  const [selectedCoord, setSelectedCoord] = useState<{ lat: number; lon: number } | null>(null);
  const [activeTab, setActiveTab] = useState<'monitor' | 'regimes' | 'evals' | 'cases'>('monitor');

  const handleSelectCoordinate = (lat: number, lon: number) => {
    setSelectedCoord({ lat, lon });
  };

  return (
    <div className="min-h-screen bg-[#080C15] text-slate-100 flex flex-col font-sans selection:bg-sky-500/30 selection:text-sky-200">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 bg-[#0B0F19]/90 backdrop-blur border-b border-slate-800/80 px-4 sm:px-8 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-tr from-sky-600 to-indigo-600 rounded-xl shadow-md text-white shadow-sky-500/20">
            <CloudLightning className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-black tracking-tight text-slate-100">BustWatch</h1>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 bg-sky-500/10 text-sky-400 border border-sky-500/30 rounded-full font-bold">
                SIH26079 Operational
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Medium-Range NWP Forecast Bust Detection & Confidence Mapping</p>
          </div>
        </div>

        {/* Center Tabs */}
        <nav className="hidden md:flex items-center bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs font-medium">
          <button
            onClick={() => setActiveTab('monitor')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition-all ${
              activeTab === 'monitor'
                ? 'bg-sky-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Confidence Maps
          </button>
          <button
            onClick={() => setActiveTab('regimes')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition-all ${
              activeTab === 'regimes'
                ? 'bg-sky-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            Regime Simulator
          </button>
          <button
            onClick={() => setActiveTab('evals')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition-all ${
              activeTab === 'evals'
                ? 'bg-sky-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Reliability & Evals
          </button>
          <button
            onClick={() => setActiveTab('cases')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition-all ${
              activeTab === 'cases'
                ? 'bg-sky-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Bookmark className="w-3.5 h-3.5" />
            Case Studies
          </button>
        </nav>

        {/* Right Status */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-950/40 px-3 py-1 rounded-full border border-emerald-800/50 font-mono">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            API & Models Live
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Lead-Time Scrubber */}
        <LeadTimeSlider leadTime={leadTime} onChange={setLeadTime} />

        {/* Tab Content */}
        {activeTab === 'monitor' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Spatial Map Raster */}
              <div className="lg:col-span-7">
                <SpatialMap leadTime={leadTime} onSelectCoordinate={handleSelectCoordinate} />
              </div>

              {/* Right Station Profile */}
              <div className="lg:col-span-5">
                <StationProfile selectedCoord={selectedCoord} />
              </div>
            </div>

            {/* Quick Metrics Bar */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ReliabilityView />
              <RegimeSimulator />
            </div>

            <CaseStudies />
          </div>
        )}

        {activeTab === 'regimes' && (
          <div className="space-y-6">
            <RegimeSimulator />
            <StationProfile selectedCoord={selectedCoord} />
          </div>
        )}

        {activeTab === 'evals' && (
          <div className="space-y-6">
            <ReliabilityView />
            <CaseStudies />
          </div>
        )}

        {activeTab === 'cases' && (
          <div className="space-y-6">
            <CaseStudies />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#090D17] py-6 px-4 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            BustWatch &bull; Medium-Range NWP Forecast Bust Detection &bull; MIT License
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>FastAPI Backend</span>
            <span>&bull;</span>
            <span>Calibrated LightGBM</span>
            <span>&bull;</span>
            <span>React + Vite</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
