import React, { useState, useEffect } from 'react';
import { History, Search, FileText, Trash2, ExternalLink, RefreshCw, AlertCircle } from 'lucide-react';

export default function AuditHistoryTab() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSeverity, setSelectedSeverity] = useState('ALL');

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/history');
      if (!res.ok) throw new Error('Failed to load history');
      const data = await res.json();
      setHistory(data.inspections || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this inspection record from history?')) return;
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/history/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setHistory(history.filter((item) => item.id !== id));
      }
    } catch (err) {
      console.error(err);
    }
  };

  const filteredHistory = history.filter((item) => {
    const matchesSearch = item.filename?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          item.recommendation?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSev = selectedSeverity === 'ALL' || item.severity_level?.toUpperCase() === selectedSeverity;
    return matchesSearch && matchesSev;
  });

  return (
    <div>
      {/* Header Bar */}
      <div className="glass-panel" style={{ padding: '20px 24px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <History size={20} color="#06b6d4" />
              Inspection Audit Trail & Persistent Storage
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '2px' }}>
              SQLite inspection records with road distress metrics and printable engineering reports.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Search size={14} style={{ position: 'absolute', left: '10px', color: 'var(--text-dim)' }} />
              <input
                type="text"
                placeholder="Search audit records..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '8px 12px 8px 30px',
                  color: '#ffffff',
                  fontSize: '13px',
                  outline: 'none',
                  width: '200px',
                }}
              />
            </div>

            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '8px 12px',
                color: '#ffffff',
                fontSize: '13px',
                outline: 'none',
              }}
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical</option>
              <option value="SEVERE">Severe</option>
              <option value="MODERATE">Moderate</option>
              <option value="LOW">Low</option>
              <option value="NONE">None</option>
            </select>

            <button onClick={fetchHistory} className="btn btn-secondary" style={{ padding: '8px 12px' }}>
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </div>
      </div>

      {/* History Table */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-dim)' }}>Loading inspection records...</div>
        ) : filteredHistory.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-dim)' }}>
            No inspection records found matching criteria.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Scan #</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Capture Preview</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Timestamp</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Model Used</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Severity Level</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Potholes</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Damage Area</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Scales (S/M/L)</th>
                  <th style={{ padding: '12px', textAlign: 'right', color: 'var(--text-muted)' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredHistory.map((item) => (
                  <tr key={item.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '12px', fontWeight: 600 }}>#{item.id}</td>
                    <td style={{ padding: '12px' }}>
                      {item.thumbnail_base64 ? (
                        <img
                          src={item.thumbnail_base64}
                          alt="Thumb"
                          style={{ width: '48px', height: '48px', borderRadius: '4px', objectFit: 'cover', border: '1px solid var(--border-subtle)' }}
                        />
                      ) : (
                        <div style={{ width: '48px', height: '48px', borderRadius: '4px', background: 'var(--bg-secondary)' }} />
                      )}
                    </td>
                    <td style={{ padding: '12px', color: 'var(--text-dim)', fontSize: '12px' }}>{item.timestamp}</td>
                    <td style={{ padding: '12px', fontSize: '12px', color: '#93c5fd' }}>{item.model_used}</td>
                    <td style={{ padding: '12px' }}>
                      <span className={`severity-pill ${item.badge_color || 'blue'}`} style={{ fontSize: '11px', padding: '2px 8px' }}>
                        {item.severity_level} ({item.severity_score})
                      </span>
                    </td>
                    <td style={{ padding: '12px', fontWeight: 600 }}>{item.pothole_count}</td>
                    <td style={{ padding: '12px' }}>{item.damaged_area_percentage}%</td>
                    <td style={{ padding: '12px', fontSize: '12px' }}>
                      <span style={{ color: '#38bdf8' }}>{item.small_count}S</span> &bull;{' '}
                      <span style={{ color: '#fbbf24' }}>{item.medium_count}M</span> &bull;{' '}
                      <span style={{ color: '#f87171' }}>{item.large_count}L</span>
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                        <a
                          href={`http://127.0.0.1:8000/api/report/${item.id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="btn btn-secondary"
                          style={{ padding: '6px 10px', fontSize: '12px', textDecoration: 'none' }}
                        >
                          <FileText size={13} /> Report
                        </a>
                        <button
                          onClick={() => handleDelete(item.id)}
                          className="btn btn-danger"
                          style={{ padding: '6px 10px', fontSize: '12px' }}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
