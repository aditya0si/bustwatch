import React, { useState, useEffect } from 'react';
import { MapPin, Layers } from 'lucide-react';
import { GridRasterResponse, fetchConfidenceGrid } from '../services/api';

interface SpatialMapProps {
  leadTime: number;
  onSelectCoordinate: (lat: number, lon: number) => void;
}

export const SpatialMap: React.FC<SpatialMapProps> = ({ leadTime, onSelectCoordinate }) => {
  const [gridData, setGridData] = useState<GridRasterResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeLayer, setActiveLayer] = useState<'probability' | 'confidence' | 'error'>('probability');
  const [hoveredCell, setHoveredCell] = useState<{
    lat: number;
    lon: number;
    prob: number;
    conf: number;
    error: number;
  } | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchConfidenceGrid(leadTime)
      .then((data) => {
        if (isMounted) {
          setGridData(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error('Error loading grid:', err);
        setLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [leadTime]);

  const getColorForValue = (val: number, type: 'probability' | 'confidence' | 'error') => {
    if (type === 'probability') {
      if (val < 0.15) return 'rgba(16, 185, 129, 0.75)'; // Emerald
      if (val < 0.35) return 'rgba(245, 158, 11, 0.75)'; // Amber
      if (val < 0.60) return 'rgba(249, 115, 22, 0.85)'; // Orange
      return 'rgba(239, 68, 68, 0.90)'; // Red (Bust)
    } else if (type === 'confidence') {
      if (val > 0.85) return 'rgba(16, 185, 129, 0.8)';
      if (val > 0.65) return 'rgba(245, 158, 11, 0.8)';
      if (val > 0.40) return 'rgba(249, 115, 22, 0.85)';
      return 'rgba(239, 68, 68, 0.9)';
    } else {
      // Error in meters Z500
      if (val < 30) return 'rgba(56, 189, 248, 0.7)';
      if (val < 60) return 'rgba(234, 179, 8, 0.8)';
      if (val < 100) return 'rgba(249, 115, 22, 0.85)';
      return 'rgba(225, 29, 72, 0.9)';
    }
  };

  return (
    <div className="bg-[#10172A] border border-slate-800/80 rounded-xl p-5 shadow-lg flex flex-col h-full">
      {/* Header & Layer Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-sky-400" />
            <h3 className="text-base font-semibold text-slate-100">Spatio-Temporal NWP Confidence Map</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Calibrated regional bust risk & 500hPa height error distributions across Day {leadTime}
          </p>
        </div>

        {/* Layer Buttons */}
        <div className="flex items-center bg-slate-900/80 p-1 rounded-lg border border-slate-800 self-start">
          <button
            onClick={() => setActiveLayer('probability')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeLayer === 'probability'
                ? 'bg-sky-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Bust Risk
          </button>
          <button
            onClick={() => setActiveLayer('confidence')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeLayer === 'confidence'
                ? 'bg-sky-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Confidence Score
          </button>
          <button
            onClick={() => setActiveLayer('error')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeLayer === 'error'
                ? 'bg-sky-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Expected Z500 Error
          </button>
        </div>
      </div>

      {/* Map Raster Display */}
      <div className="relative flex-1 min-h-[360px] bg-[#0A0E1A] rounded-xl border border-slate-800/60 overflow-hidden flex flex-col justify-center items-center p-3">
        {loading ? (
          <div className="flex flex-col items-center gap-3 text-slate-400">
            <div className="w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs font-mono">Rendering Day {leadTime} spatial raster...</span>
          </div>
        ) : gridData ? (
          <div className="w-full h-full flex flex-col justify-between">
            {/* 2D Gridded Raster Visualizer */}
            <div className="grid gap-1 w-full h-full p-2" style={{
              gridTemplateRows: `repeat(${gridData.shape[0]}, minmax(0, 1fr))`,
            }}>
              {gridData.latitudes.map((lat, i) => (
                <div key={lat} className="grid gap-1 w-full" style={{
                  gridTemplateColumns: `repeat(${gridData.shape[1]}, minmax(0, 1fr))`
                }}>
                  {gridData.longitudes.map((lon, j) => {
                    const prob = gridData.calibrated_bust_probability[i][j];
                    const conf = gridData.confidence_score[i][j];
                    const err = gridData.expected_error_z500_m[i][j];

                    const displayVal =
                      activeLayer === 'probability'
                        ? prob
                        : activeLayer === 'confidence'
                        ? conf
                        : err;

                    const bg = getColorForValue(displayVal, activeLayer);

                    return (
                      <div
                        key={`${lat}-${lon}`}
                        onClick={() => onSelectCoordinate(lat, lon)}
                        onMouseEnter={() => setHoveredCell({ lat, lon, prob, conf, error: err })}
                        onMouseLeave={() => setHoveredCell(null)}
                        style={{ backgroundColor: bg }}
                        className="rounded-sm transition-transform duration-100 hover:scale-110 hover:z-20 cursor-pointer border border-black/20 relative group"
                      />
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Hover Tooltip Overlay */}
            {hoveredCell && (
              <div className="absolute bottom-4 left-4 bg-slate-900/95 border border-sky-500/40 rounded-lg p-3 shadow-2xl text-xs font-mono space-y-1 backdrop-blur z-30 pointer-events-none">
                <div className="text-sky-400 font-bold flex items-center gap-1.5">
                  <MapPin className="w-3.5 h-3.5" />
                  Lat: {hoveredCell.lat.toFixed(1)}°, Lon: {hoveredCell.lon.toFixed(1)}°
                </div>
                <div className="text-slate-300">
                  Bust Probability: <span className="font-bold text-amber-400">{(hoveredCell.prob * 100).toFixed(1)}%</span>
                </div>
                <div className="text-slate-300">
                  Confidence Score: <span className="font-bold text-emerald-400">{(hoveredCell.conf * 100).toFixed(1)}%</span>
                </div>
                <div className="text-slate-300">
                  Expected Z500 Error: <span className="font-bold text-rose-400">{hoveredCell.error.toFixed(1)} m</span>
                </div>
                <div className="text-[10px] text-sky-400/80 pt-0.5">Click to view 10-day decay curve</div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-slate-500 text-xs">No spatial grid data available.</div>
        )}
      </div>

      {/* Map Legend */}
      <div className="flex flex-wrap items-center justify-between gap-4 mt-3 pt-3 border-t border-slate-800 text-xs text-slate-400">
        <div className="flex items-center gap-4">
          <span className="font-semibold text-slate-300">Risk Scale:</span>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-emerald-500 inline-block" />
            <span>Low (&lt;15%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-amber-500 inline-block" />
            <span>Moderate (15–35%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-orange-500 inline-block" />
            <span>Elevated (35–60%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-rose-500 inline-block" />
            <span>Critical Bust (&gt;60%)</span>
          </div>
        </div>
        <div className="text-[11px] text-slate-500">
          Grid: 2.5° × 2.5° Resolution (GFS/ECMWF domain)
        </div>
      </div>
    </div>
  );
};
