import React, { useState, useRef } from 'react';
import { Eye, Sliders, Layers, Columns } from 'lucide-react';

export default function ImageViewer({ originalBase64, overlayBase64, instances = [], selectedInstanceId, onSelectInstance }) {
  const [viewMode, setViewMode] = useState('slider'); // 'slider' | 'overlay' | 'original' | 'side-by-side'
  const [sliderPosition, setSliderPosition] = useState(50); // percentage 0 - 100
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);

  if (!originalBase64) {
    return (
      <div className="glass-panel" style={{ height: '440px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)' }}>
        Select a sample or upload a road photograph to run multi-scale inspection.
      </div>
    );
  }

  const handleMouseMove = (e) => {
    if (!isDragging || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSliderPosition((x / rect.width) * 100);
  };

  const handleTouchMove = (e) => {
    if (!containerRef.current) return;
    const touch = e.touches[0];
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(touch.clientX - rect.left, rect.width));
    setSliderPosition((x / rect.width) * 100);
  };

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      {/* Control Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', gap: '6px', background: 'var(--bg-secondary)', padding: '4px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
          <button
            onClick={() => setViewMode('slider')}
            className={`btn ${viewMode === 'slider' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 12px', fontSize: '12px' }}
          >
            <Sliders size={14} /> Split Slider
          </button>
          <button
            onClick={() => setViewMode('overlay')}
            className={`btn ${viewMode === 'overlay' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 12px', fontSize: '12px' }}
          >
            <Layers size={14} /> Overlay
          </button>
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`btn ${viewMode === 'side-by-side' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 12px', fontSize: '12px' }}
          >
            <Columns size={14} /> Side-by-Side
          </button>
          <button
            onClick={() => setViewMode('original')}
            className={`btn ${viewMode === 'original' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 12px', fontSize: '12px' }}
          >
            <Eye size={14} /> Original
          </button>
        </div>

        <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
          {viewMode === 'slider' && 'Drag the vertical divider to compare raw vs segmented road'}
        </div>
      </div>

      {/* Main Image Stage */}
      {viewMode === 'side-by-side' ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          <div>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px' }}>Original Road Surface</div>
            <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-subtle)', background: '#000' }}>
              <img src={originalBase64} alt="Original" style={{ width: '100%', height: '380px', objectFit: 'contain', display: 'block' }} />
            </div>
          </div>
          <div>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#f87171', marginBottom: '6px' }}>MultiScale-RoadNet Detection & Segmentation</div>
            <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-subtle)', background: '#000' }}>
              <img src={overlayBase64 || originalBase64} alt="Overlay" style={{ width: '100%', height: '380px', objectFit: 'contain', display: 'block' }} />
            </div>
          </div>
        </div>
      ) : (
        <div
          ref={containerRef}
          onMouseDown={() => setIsDragging(true)}
          onMouseUp={() => setIsDragging(false)}
          onMouseLeave={() => setIsDragging(false)}
          onMouseMove={handleMouseMove}
          onTouchMove={handleTouchMove}
          style={{
            position: 'relative',
            width: '100%',
            height: '420px',
            borderRadius: 'var(--radius-md)',
            overflow: 'hidden',
            background: '#030508',
            border: '1px solid var(--border-subtle)',
            userSelect: 'none',
            cursor: viewMode === 'slider' ? 'ew-resize' : 'default',
          }}
        >
          {/* Base Layer: Overlay (or Original if original mode) */}
          <img
            src={viewMode === 'original' ? originalBase64 : (overlayBase64 || originalBase64)}
            alt="Base Layer"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
            }}
          />

          {/* Reveal Layer: Original Road (Clipped by slider position) */}
          {viewMode === 'slider' && overlayBase64 && (
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: `${sliderPosition}%`,
                height: '100%',
                overflow: 'hidden',
                borderRight: '2px solid #ffffff',
                boxShadow: '0 0 16px rgba(0,0,0,0.8)',
              }}
            >
              <img
                src={originalBase64}
                alt="Original Reveal"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%',
                  height: '100%',
                  objectFit: 'contain',
                }}
              />
            </div>
          )}

          {/* Slider Handle */}
          {viewMode === 'slider' && overlayBase64 && (
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: `${sliderPosition}%`,
                transform: 'translate(-50%, -50%)',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: '#ffffff',
                color: '#0f172a',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 10px rgba(0,0,0,0.6)',
                pointerEvents: 'none',
                fontSize: '12px',
                fontWeight: 800,
              }}
            >
              &#8644;
            </div>
          )}

          {/* Chip labels on corners */}
          {viewMode === 'slider' && (
            <>
              <div style={{ position: 'absolute', top: 12, left: 12, background: 'rgba(0,0,0,0.6)', padding: '4px 10px', borderRadius: '4px', fontSize: '11px', color: '#cbd5e1' }}>
                Original Surface
              </div>
              <div style={{ position: 'absolute', top: 12, right: 12, background: 'rgba(239, 68, 68, 0.75)', padding: '4px 10px', borderRadius: '4px', fontSize: '11px', color: '#ffffff', fontWeight: 600 }}>
                MultiScale Segmentation
              </div>
            </>
          )}
        </div>
      )}

      {/* Instance Chip Row */}
      {instances.length > 0 && (
        <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '8px', overflowX: 'auto', paddingBottom: '4px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>
            Detected Potholes ({instances.length}):
          </span>
          {instances.map((inst) => {
            const isSelected = selectedInstanceId === inst.id;
            const sc = inst.scale;
            const badgeBg = sc === 'small' ? '#0284c7' : (sc === 'medium' ? '#d97706' : '#dc2626');
            return (
              <button
                key={inst.id}
                onClick={() => onSelectInstance && onSelectInstance(isSelected ? null : inst.id)}
                style={{
                  background: isSelected ? badgeBg : 'var(--bg-secondary)',
                  color: '#ffffff',
                  border: `1px solid ${isSelected ? '#ffffff' : 'var(--border-subtle)'}`,
                  padding: '4px 10px',
                  borderRadius: '16px',
                  fontSize: '12px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  whiteSpace: 'nowrap',
                  transition: 'all 0.15s ease'
                }}
              >
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: badgeBg }} />
                <span>#{inst.id} {inst.scale.toUpperCase()} ({inst.area_px} px)</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
