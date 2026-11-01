import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, Wrench, Activity } from 'lucide-react';

export default function SeverityGauge({ assessment, onGenerateReport, isGeneratingReport }) {
  if (!assessment) return null;

  const {
    severity_score = 0,
    severity_level = 'None',
    badge_color = 'emerald',
    action_code = 'SAFE',
    recommendation = '',
    pothole_count = 0,
    damaged_area_percentage = 0,
    scale_distribution = { small: 0, medium: 0, large: 0 },
    max_depth_proxy = 0,
  } = assessment;

  // SVG Gauge calculations
  const radius = 58;
  const stroke = 10;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (severity_score / 100) * circumference;

  const colorMap = {
    emerald: '#10b981',
    blue: '#3b82f6',
    amber: '#f59e0b',
    orange: '#f97316',
    rose: '#ef4444',
  };
  const strokeColor = colorMap[badge_color] || '#3b82f6';

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} color={strokeColor} />
            Civil Damage Severity Assessment
          </h3>
          <p style={{ fontSize: '12px', color: 'var(--text-dim)', marginTop: '2px' }}>
            ASTM D6433 Pavement Distress Index Formulation
          </p>
        </div>
        <span className={`severity-pill ${badge_color}`}>
          {severity_level} Risk
        </span>
      </div>

      {/* Main Metric Visualizer */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '28px', marginBottom: '24px' }}>
        {/* Circular Radial Gauge */}
        <div style={{ position: 'relative', width: '130px', height: '130px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg height="130" width="130" style={{ transform: 'rotate(-90deg)' }}>
            <circle
              stroke="rgba(255,255,255,0.08)"
              fill="transparent"
              strokeWidth={stroke}
              r={normalizedRadius}
              cx="65"
              cy="65"
            />
            <circle
              stroke={strokeColor}
              fill="transparent"
              strokeWidth={stroke}
              strokeDasharray={circumference + ' ' + circumference}
              style={{ strokeDashoffset, transition: 'stroke-dashoffset 0.8s ease' }}
              strokeLinecap="round"
              r={normalizedRadius}
              cx="65"
              cy="65"
            />
          </svg>
          <div style={{ position: 'absolute', textAlign: 'center' }}>
            <div style={{ fontSize: '28px', fontWeight: 800, color: '#ffffff', lineHeight: 1 }}>
              {severity_score}
            </div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', marginTop: '2px' }}>
              Index / 100
            </div>
          </div>
        </div>

        {/* Quick Summary Cards */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Potholes Detected</div>
            <div style={{ fontSize: '20px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>{pothole_count}</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Surface Damage</div>
            <div style={{ fontSize: '20px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>{damaged_area_percentage}%</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Scale Distribution</div>
            <div style={{ fontSize: '13px', fontWeight: 600, marginTop: '6px', display: 'flex', gap: '6px' }}>
              <span style={{ color: '#38bdf8' }}>{scale_distribution.small}S</span>
              <span style={{ color: 'var(--text-dim)' }}>|</span>
              <span style={{ color: '#fbbf24' }}>{scale_distribution.medium}M</span>
              <span style={{ color: 'var(--text-dim)' }}>|</span>
              <span style={{ color: '#f87171' }}>{scale_distribution.large}L</span>
            </div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Depth Proxy</div>
            <div style={{ fontSize: '20px', fontWeight: 700, color: strokeColor, marginTop: '4px' }}>{max_depth_proxy}</div>
          </div>
        </div>
      </div>

      {/* Engineering Recommendation Box */}
      <div style={{
        background: 'rgba(255,255,255,0.02)',
        borderLeft: `4px solid ${strokeColor}`,
        padding: '14px 18px',
        borderRadius: 'var(--radius-sm)',
        marginBottom: '20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
          <span style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px', color: strokeColor, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Wrench size={14} /> Recommended Action: {action_code}
          </span>
        </div>
        <p style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.45' }}>
          {recommendation}
        </p>
      </div>

      {/* Downloadable Inspection Report Button */}
      {onGenerateReport && (
        <button
          onClick={onGenerateReport}
          disabled={isGeneratingReport}
          className="btn btn-secondary"
          style={{ width: '100%', justifyContent: 'center' }}
        >
          {isGeneratingReport ? 'Generating Audit Report...' : 'Download Civil Inspection Audit Report (HTML/PDF)'}
        </button>
      )}
    </div>
  );
}
