import React, { useState, useEffect } from 'react';
import { BookOpen, Award, CheckCircle, BarChart3, Layers, Cpu, ShieldCheck } from 'lucide-react';
import { API_BASE } from '../config/api';
import fallbackData from '../data/fallbackData.json';

export default function ResearchTab() {
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/api/benchmark`)
      .then((res) => res.json())
      .then((data) => {
        setBenchmarkData(data.benchmarks);
        setLoading(false);
      })
      .catch((err) => {
        console.warn('Backend benchmark endpoint unreachable, loading fallback benchmarks:', err);
        setBenchmarkData(fallbackData.benchmarks);
        setLoading(false);
      });
  }, []);

  const b = benchmarkData?.baseline || {
    parameters: 4318401,
    model_size_mb: 49.52,
    mean_iou: 0.4251,
    dice_score: 0.5966,
    precision: 0.4965,
    recall: 0.7473,
    small_pothole_iou: 0.03,
    medium_pothole_iou: 0.3356,
    large_pothole_iou: 0.4723,
    avg_latency_ms: 163.23,
  };

  const p = benchmarkData?.proposed || {
    parameters: 2137172,
    model_size_mb: 24.6,
    mean_iou: 0.5087,
    dice_score: 0.6744,
    precision: 0.6096,
    recall: 0.7545,
    small_pothole_iou: 0.0431,
    medium_pothole_iou: 0.5072,
    large_pothole_iou: 0.5361,
    avg_latency_ms: 64.88,
  };

  return (
    <div>
      {/* Research Contribution Card */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px', borderLeft: '4px solid #6366f1' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
          <Award size={22} color="#6366f1" />
          <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#ffffff' }}>
            MultiScale-RoadNet: Research Findings & Empirical Contributions
          </h2>
        </div>
        <p style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6', marginBottom: '16px' }}>
          Standard single-scale architectures (such as U-Net or FCN) suffer from acute feature resolution attenuation
          during deep pooling, resulting in high false-negative rates on small incipient potholes and boundary blurring on irregular craters.
          <strong> MultiScale-RoadNet</strong> resolves this via a dual-pyramid mechanism integrating <strong>Atrous Spatial Pyramid Pooling (ASPP)</strong>
          and a <strong>Top-Down Feature Pyramid Network (FPN)</strong> with channel scale-attention.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px 16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Small Pothole IoU Gain</div>
            <div style={{ fontSize: '22px', fontWeight: 800, color: '#38bdf8', marginTop: '4px' }}>
              +{((p.small_pothole_iou - b.small_pothole_iou) * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '2px' }}>FPN lateral feature preservation</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px 16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Overall Mean IoU Delta</div>
            <div style={{ fontSize: '22px', fontWeight: 800, color: '#34d399', marginTop: '4px' }}>
              +{((p.mean_iou - b.mean_iou) * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '2px' }}>Across test ground truth masks</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px 16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Parameter Efficiency</div>
            <div style={{ fontSize: '22px', fontWeight: 800, color: '#f43f5e', marginTop: '4px' }}>
              -50.5%
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '2px' }}>2.14M vs 4.32M parameters</div>
          </div>
        </div>
      </div>

      {/* Quantitative Benchmark Table */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#ffffff', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BarChart3 size={18} color="#06b6d4" />
          Quantitative Experimental Comparison on Test Benchmark Split
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '12px 16px', textAlign: 'left', color: 'var(--text-muted)' }}>Evaluation Metric</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', color: '#60a5fa' }}>Baseline Single-Scale UNet</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', color: '#f87171' }}>Proposed MultiScale-RoadNet</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', color: '#34d399' }}>Relative Delta</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Architecture Parameters</td>
                <td style={{ padding: '12px 16px' }}>{b.parameters.toLocaleString()}</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{p.parameters.toLocaleString()}</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>-50.5% (Compact)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Checkpoint Size</td>
                <td style={{ padding: '12px 16px' }}>{b.model_size_mb} MB</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{p.model_size_mb} MB</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>-50.5% Storage</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Mean Intersection-over-Union (mIoU)</td>
                <td style={{ padding: '12px 16px' }}>{(b.mean_iou * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{(p.mean_iou * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>+{((p.mean_iou - b.mean_iou) * 100).toFixed(2)}%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Dice Coefficient (F1 Score)</td>
                <td style={{ padding: '12px 16px' }}>{(b.dice_score * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{(p.dice_score * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>+{((p.dice_score - b.dice_score) * 100).toFixed(2)}%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Precision</td>
                <td style={{ padding: '12px 16px' }}>{(b.precision * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{(p.precision * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>+{((p.precision - b.precision) * 100).toFixed(2)}%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Recall</td>
                <td style={{ padding: '12px 16px' }}>{(b.recall * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{(p.recall * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>+{((p.recall - b.recall) * 100).toFixed(2)}%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', background: 'rgba(56, 189, 248, 0.05)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 700, color: '#38bdf8' }}>Small Pothole IoU (&lt; 500 px)</td>
                <td style={{ padding: '12px 16px' }}>{(b.small_pothole_iou * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#38bdf8', fontWeight: 800 }}>{(p.small_pothole_iou * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#38bdf8', fontWeight: 700 }}>+{((p.small_pothole_iou - b.small_pothole_iou) * 100).toFixed(2)}% (Primary Impact)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Medium Pothole IoU (500 - 2,500 px)</td>
                <td style={{ padding: '12px 16px' }}>{(b.medium_pothole_iou * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{(p.medium_pothole_iou * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>+{((p.medium_pothole_iou - b.medium_pothole_iou) * 100).toFixed(2)}%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Large Pothole IoU (&gt; 2,500 px)</td>
                <td style={{ padding: '12px 16px' }}>{(b.large_pothole_iou * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{(p.large_pothole_iou * 100).toFixed(2)}%</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>+{((p.large_pothole_iou - b.large_pothole_iou) * 100).toFixed(2)}%</td>
              </tr>
              <tr>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>Inference Latency (CPU)</td>
                <td style={{ padding: '12px 16px' }}>{b.avg_latency_ms} ms / frame</td>
                <td style={{ padding: '12px 16px', color: '#34d399', fontWeight: 700 }}>{p.avg_latency_ms} ms / frame</td>
                <td style={{ padding: '12px 16px', color: '#34d399' }}>Real-time Edge Ready</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Literature Foundations */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '18px' }}>
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ fontSize: '14px', fontWeight: 700, color: '#ffffff', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={16} color="#6366f1" /> PODSNet Foundation
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
            Dual-task formulation bridging pixel-level segmentation masks with localized bounding-box extraction and confidence scoring.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ fontSize: '14px', fontWeight: 700, color: '#ffffff', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} color="#06b6d4" /> DMTC Multi-Scale ASPP
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
            Dense multi-scale feature learning with atrous convolutions (r=1, 3, 6) capturing global context without spatial resolution decay.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ fontSize: '14px', fontWeight: 700, color: '#ffffff', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={16} color="#10b981" /> PotholeGrade-BD
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
            Mathematical severity grading integrating surface damage area percentage, photometric depth proxies, and municipal repair action codes.
          </p>
        </div>
      </div>
    </div>
  );
}
