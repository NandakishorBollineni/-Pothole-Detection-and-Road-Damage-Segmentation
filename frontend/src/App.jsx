import React, { useState, useEffect } from 'react';
import {
  Layers,
  Columns,
  BarChart3,
  History,
  Upload,
  Sparkles,
  Camera,
  Activity,
  Shield,
  FileCheck,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';

import SeverityGauge from './components/SeverityGauge';
import ImageViewer from './components/ImageViewer';
import ModelComparisonTab from './components/ModelComparisonTab';
import ResearchTab from './components/ResearchTab';
import AuditHistoryTab from './components/AuditHistoryTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('inspector'); // 'inspector' | 'comparison' | 'research' | 'history'
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState('');
  const [modelChoice, setModelChoice] = useState('multiscale');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isInspecting, setIsInspecting] = useState(false);
  const [inspectionResult, setInspectionResult] = useState(null);
  const [selectedInstanceId, setSelectedInstanceId] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Load samples and health
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/health')
      .then((res) => res.json())
      .then((data) => setHealthStatus(data))
      .catch((err) => console.log('Backend not connected yet', err));

    fetch('http://127.0.0.1:8000/api/samples')
      .then((res) => res.json())
      .then((data) => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
          setSelectedSample(data.samples[0].id);
          // Auto run initial inspection on first sample
          triggerInspection(data.samples[0].id, null, 'multiscale');
        }
      })
      .catch((err) => console.log('Samples fetch error', err));
  }, []);

  const triggerInspection = async (sampleName, fileObj, model) => {
    setIsInspecting(true);
    setErrorMsg(null);
    try {
      let url = `http://127.0.0.1:8000/api/predict?model_choice=${model || modelChoice}`;
      let options = { method: 'POST' };

      if (fileObj) {
        const formData = new FormData();
        formData.append('file', fileObj);
        options.body = formData;
      } else if (sampleName) {
        url += `&sample_name=${encodeURIComponent(sampleName)}`;
      }

      const res = await fetch(url, options);
      if (!res.ok) throw new Error('Inspection API request failed');
      const data = await res.json();
      setInspectionResult(data);
    } catch (err) {
      console.error(err);
      setErrorMsg('Failed to run road damage inspection. Ensure FastAPI backend is running on port 8000.');
    } finally {
      setIsInspecting(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      setSelectedSample('');
      triggerInspection(null, file, modelChoice);
    }
  };

  const handleGenerateReport = () => {
    if (inspectionResult?.inspection_id) {
      window.open(`http://127.0.0.1:8000/api/report/${inspectionResult.inspection_id}`, '_blank');
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navbar */}
      <header className="app-header">
        <div className="header-container">
          <div className="brand-wrapper">
            <div className="brand-icon">
              <Shield size={22} color="#ffffff" />
            </div>
            <div>
              <div className="brand-title">PotholeVision AI</div>
              <div className="brand-subtitle">MultiScale Deep Learning Road Damage Inspector</div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="nav-tabs">
            <button
              onClick={() => setActiveTab('inspector')}
              className={`nav-tab ${activeTab === 'inspector' ? 'active' : ''}`}
            >
              <Camera size={15} /> Inspector
            </button>
            <button
              onClick={() => setActiveTab('comparison')}
              className={`nav-tab ${activeTab === 'comparison' ? 'active' : ''}`}
            >
              <Columns size={15} /> Dual Model Comparison
            </button>
            <button
              onClick={() => setActiveTab('research')}
              className={`nav-tab ${activeTab === 'research' ? 'active' : ''}`}
            >
              <BarChart3 size={15} /> Research & Benchmarks
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`nav-tab ${activeTab === 'history' ? 'active' : ''}`}
            >
              <History size={15} /> Audit History
            </button>
          </nav>
        </div>
      </header>

      {/* Main Body */}
      <main className="main-content" style={{ flex: 1 }}>
        {errorMsg && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.35)',
            borderRadius: 'var(--radius-md)',
            padding: '12px 18px',
            color: '#fca5a5',
            marginBottom: '20px',
            fontSize: '13px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <AlertTriangle size={18} color="#ef4444" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* TAB 1: INSPECTOR */}
        {activeTab === 'inspector' && (
          <div>
            {/* Top Toolbar: Sample selector & upload & model picker */}
            <div className="glass-panel" style={{ padding: '16px 20px', marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                {/* Left: Input Selection */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)' }}>Road Image:</span>
                  
                  {/* Benchmark Samples Dropdown */}
                  <select
                    value={selectedSample}
                    onChange={(e) => {
                      setSelectedSample(e.target.value);
                      setUploadedFile(null);
                      triggerInspection(e.target.value, null, modelChoice);
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
                    <option value="" disabled>-- Select Benchmark Sample --</option>
                    {samples.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name}
                      </option>
                    ))}
                  </select>

                  <span style={{ color: 'var(--text-dim)', fontSize: '12px' }}>or</span>

                  {/* Upload Button */}
                  <label className="btn btn-secondary" style={{ padding: '8px 14px', fontSize: '13px', cursor: 'pointer' }}>
                    <Upload size={14} /> Upload Custom Road Image
                    <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
                  </label>

                  {uploadedFile && (
                    <span style={{ fontSize: '12px', color: '#38bdf8' }}>
                      File: {uploadedFile.name}
                    </span>
                  )}
                </div>

                {/* Right: Architecture Selector */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Architecture:</span>
                  <div style={{ display: 'flex', background: 'var(--bg-secondary)', padding: '3px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    <button
                      onClick={() => {
                        setModelChoice('multiscale');
                        triggerInspection(selectedSample, uploadedFile, 'multiscale');
                      }}
                      style={{
                        background: modelChoice === 'multiscale' ? 'var(--accent-indigo)' : 'transparent',
                        color: modelChoice === 'multiscale' ? '#ffffff' : 'var(--text-muted)',
                        border: 'none',
                        padding: '6px 12px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'all 0.15s ease'
                      }}
                    >
                      MultiScale-RoadNet (Proposed)
                    </button>
                    <button
                      onClick={() => {
                        setModelChoice('baseline');
                        triggerInspection(selectedSample, uploadedFile, 'baseline');
                      }}
                      style={{
                        background: modelChoice === 'baseline' ? '#3b82f6' : 'transparent',
                        color: modelChoice === 'baseline' ? '#ffffff' : 'var(--text-muted)',
                        border: 'none',
                        padding: '6px 12px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'all 0.15s ease'
                      }}
                    >
                      Baseline UNet
                    </button>
                  </div>

                  <button
                    onClick={() => triggerInspection(selectedSample, uploadedFile, modelChoice)}
                    disabled={isInspecting}
                    className="btn btn-primary"
                    style={{ padding: '8px 16px', fontSize: '13px' }}
                  >
                    {isInspecting ? (
                      <>
                        <Sparkles size={14} className="animate-spin" /> Scanning Pavement...
                      </>
                    ) : (
                      <>
                        <Sparkles size={14} /> Run Inspection
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Main Stage Grid: Interactive Image Viewer & Severity Gauge */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '20px' }}>
              <div>
                <ImageViewer
                  originalBase64={inspectionResult?.original_image_base64}
                  overlayBase64={inspectionResult?.overlay_image_base64}
                  instances={inspectionResult?.assessment?.instances || []}
                  selectedInstanceId={selectedInstanceId}
                  onSelectInstance={setSelectedInstanceId}
                />
              </div>

              <div>
                <SeverityGauge
                  assessment={inspectionResult?.assessment}
                  onGenerateReport={handleGenerateReport}
                  isGeneratingReport={false}
                />
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: DUAL MODEL COMPARISON */}
        {activeTab === 'comparison' && (
          <ModelComparisonTab
            samples={samples}
            selectedSample={selectedSample}
            onSelectSample={setSelectedSample}
          />
        )}

        {/* TAB 3: RESEARCH & BENCHMARKS */}
        {activeTab === 'research' && <ResearchTab />}

        {/* TAB 4: AUDIT HISTORY */}
        {activeTab === 'history' && <AuditHistoryTab />}
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '16px 24px',
        background: 'rgba(7, 9, 14, 0.95)',
        fontSize: '12px',
        color: 'var(--text-dim)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        maxWidth: '1400px',
        margin: '0 auto',
        width: '100%'
      }}>
        <div>
          MultiScale-RoadNet &bull; Multi-Scale Deep Learning for Pothole Detection and Road Damage Segmentation
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <span>Research Foundations: PODSNet, DMTC, PotholeGrade-BD</span>
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            color: '#34d399',
            fontSize: '11px',
            background: 'rgba(16, 185, 129, 0.1)',
            padding: '2px 8px',
            borderRadius: '12px'
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#34d399' }} />
            PyTorch CPU Active
          </span>
        </div>
      </footer>
    </div>
  );
}
