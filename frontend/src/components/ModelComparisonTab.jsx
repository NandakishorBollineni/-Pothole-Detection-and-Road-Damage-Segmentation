import React, { useState } from 'react';
import { Columns, ArrowRight, Zap, CheckCircle2, AlertCircle } from 'lucide-react';

export default function ModelComparisonTab({ samples = [], selectedSample, onSelectSample }) {
  const [comparisonData, setComparisonData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const runComparison = async (sampleName) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/compare?sample_name=${encodeURIComponent(sampleName)}`, {
        method: 'POST',
      });
      if (!res.ok) throw new Error('Comparison API error');
      const data = await res.json();
      setComparisonData(data);
    } catch (err) {
      setError('Failed to run model comparison. Please ensure backend server is active.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      {/* Header Banner */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h2 style={{ fontSize: '20px', fontWeight: 800, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Columns size={22} color="#6366f1" />
              Head-to-Head Model Comparison: Baseline UNet vs MultiScale-RoadNet
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Demonstrates how multi-scale feature pyramids (FPN) and atrous pooling (ASPP) overcome single-scale boundary loss.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Select Test Bench:</span>
            <select
              value={selectedSample}
              onChange={(e) => {
                onSelectSample(e.target.value);
                runComparison(e.target.value);
              }}
              style={{
                background: 'var(--bg-secondary)',
                color: '#ffffff',
                border: '1px solid var(--border-subtle)',
                padding: '8px 12px',
                borderRadius: 'var(--radius-sm)',
                fontSize: '13px',
                outline: 'none',
              }}
            >
              {samples.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <button
              onClick={() => selectedSample && runComparison(selectedSample)}
              disabled={isLoading || !selectedSample}
              className="btn btn-primary"
              style={{ padding: '8px 16px', fontSize: '13px' }}
            >
              {isLoading ? 'Running Dual Models...' : 'Run Comparison'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '12px 16px', borderRadius: 'var(--radius-sm)', color: '#fca5a5', marginBottom: '20px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {/* Comparison Viewports */}
      {comparisonData ? (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '18px', marginBottom: '24px' }}>
            {/* 1. Original Pavement */}
            <div className="glass-panel" style={{ padding: '16px' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: '#f8fafc', marginBottom: '10px' }}>
                1. Original Pavement
              </div>
              <div style={{ borderRadius: 'var(--radius-sm)', overflow: 'hidden', height: '280px', background: '#000' }}>
                <img src={comparisonData.original_base64} alt="Original" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              </div>
              <div style={{ marginTop: '12px', fontSize: '12px', color: 'var(--text-dim)', textAlign: 'center' }}>
                Raw Ground Surface Inspection Frame
              </div>
            </div>

            {/* 2. Baseline UNet */}
            <div className="glass-panel" style={{ padding: '16px', borderTop: '3px solid #3b82f6' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ fontSize: '13px', fontWeight: 700, color: '#60a5fa' }}>
                  2. Baseline UNet
                </div>
                <span style={{ fontSize: '11px', background: 'rgba(59, 130, 246, 0.15)', color: '#93c5fd', padding: '2px 8px', borderRadius: '4px' }}>
                  Single-Scale
                </span>
              </div>
              <div style={{ borderRadius: 'var(--radius-sm)', overflow: 'hidden', height: '280px', background: '#000' }}>
                <img src={comparisonData.baseline.overlay_base64} alt="Baseline" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              </div>
              <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-around', fontSize: '12px' }}>
                <div><strong>Defects:</strong> {comparisonData.baseline.assessment.pothole_count}</div>
                <div><strong>Damaged Area:</strong> {comparisonData.baseline.assessment.damaged_area_percentage}%</div>
                <div><strong>DSI:</strong> {comparisonData.baseline.assessment.severity_score}/100</div>
              </div>
            </div>

            {/* 3. Proposed MultiScale-RoadNet */}
            <div className="glass-panel" style={{ padding: '16px', borderTop: '3px solid #ef4444' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ fontSize: '13px', fontWeight: 700, color: '#f87171' }}>
                  3. MultiScale-RoadNet
                </div>
                <span style={{ fontSize: '11px', background: 'rgba(239, 68, 68, 0.15)', color: '#fca5a5', padding: '2px 8px', borderRadius: '4px', fontWeight: 600 }}>
                  ASPP + FPN Multi-Scale
                </span>
              </div>
              <div style={{ borderRadius: 'var(--radius-sm)', overflow: 'hidden', height: '280px', background: '#000' }}>
                <img src={comparisonData.multiscale.overlay_base64} alt="MultiScale" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              </div>
              <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-around', fontSize: '12px' }}>
                <div><strong>Defects:</strong> {comparisonData.multiscale.assessment.pothole_count}</div>
                <div><strong>Damaged Area:</strong> {comparisonData.multiscale.assessment.damaged_area_percentage}%</div>
                <div><strong>DSI:</strong> {comparisonData.multiscale.assessment.severity_score}/100</div>
              </div>
            </div>
          </div>

          {/* Research Insight Card */}
          <div className="glass-panel" style={{ padding: '20px', borderLeft: '4px solid #6366f1' }}>
            <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#ffffff', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Zap size={16} color="#6366f1" />
              Empirical Comparative Finding
            </h4>
            <p style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.5' }}>
              {comparisonData.comparison_analysis.verdict}
            </p>
          </div>
        </div>
      ) : (
        <div className="glass-panel" style={{ height: '320px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px', color: 'var(--text-dim)' }}>
          <Columns size={36} color="var(--text-dim)" />
          <div>Select a benchmark sample and click <strong>Run Comparison</strong> to inspect dual-model performance.</div>
        </div>
      )}
    </div>
  );
}
