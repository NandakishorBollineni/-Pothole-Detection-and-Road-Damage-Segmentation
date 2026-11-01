"""
Civil Road Inspection Report Generator.
Renders comprehensive, print-ready HTML/PDF reports with road distress quantification,
visual evidence, and civil engineering rehabilitation guidance.
"""
from typing import Dict, Any
from datetime import datetime


def generate_html_report(inspection: Dict[str, Any], original_base64: str = "", overlay_base64: str = "") -> str:
    """
    Renders a modern, printable HTML civil road damage inspection report.
    """
    rec_id = inspection.get("id", "N/A")
    ts = inspection.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    severity = inspection.get("severity_score", 0)
    level = inspection.get("severity_level", "Unknown")
    badge_color = inspection.get("badge_color", "blue")
    action_code = inspection.get("action_code", "MONITOR")
    pothole_count = inspection.get("pothole_count", 0)
    damage_pct = inspection.get("damaged_area_percentage", 0.0)
    small_c = inspection.get("small_count", 0)
    med_c = inspection.get("medium_count", 0)
    large_c = inspection.get("large_count", 0)
    rec = inspection.get("recommendation", "")
    model_used = inspection.get("model_used", "MultiScale-RoadNet")
    instances = inspection.get("instances", [])
    filename = inspection.get("filename", "road_scan.jpg")

    # Color mapping
    color_hex_map = {
        "emerald": "#10b981",
        "blue": "#3b82f6",
        "amber": "#f59e0b",
        "orange": "#f97316",
        "rose": "#ef4444",
    }
    accent_hex = color_hex_map.get(badge_color, "#3b82f6")

    # Table rows for individual potholes
    instance_rows = ""
    for inst in instances:
        instance_rows += f"""
        <tr>
            <td>#{inst.get('id', '-')}</td>
            <td><code>{inst.get('bbox', [])}</code></td>
            <td>{inst.get('area_px', 0):,} px</td>
            <td><span class="scale-badge scale-{inst.get('scale', 'medium')}">{inst.get('scale', 'medium').upper()}</span></td>
            <td>{inst.get('confidence', 0.90):.1%}</td>
            <td>{inst.get('depth_proxy', 0.3):.2f}</td>
        </tr>
        """
    if not instance_rows:
        instance_rows = "<tr><td colspan='6' style='text-align:center;'>No localized pothole defects detected.</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Road Distress Audit Report - #{rec_id}</title>
    <style>
        @page {{ size: A4; margin: 20mm; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            background-color: #ffffff;
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 3px solid {accent_hex};
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .brand h1 {{
            margin: 0;
            font-size: 22px;
            color: #0f172a;
            letter-spacing: -0.5px;
        }}
        .brand p {{
            margin: 4px 0 0 0;
            font-size: 13px;
            color: #64748b;
        }}
        .meta-box {{
            text-align: right;
            font-size: 13px;
            color: #475569;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }}
        .stat-val {{
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }}
        .stat-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            font-weight: 600;
        }}
        .severity-banner {{
            background: #f1f5f9;
            border-left: 6px solid {accent_hex};
            border-radius: 6px;
            padding: 16px 20px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .severity-title {{
            font-size: 16px;
            font-weight: 700;
            color: {accent_hex};
            margin-bottom: 4px;
        }}
        .severity-desc {{
            font-size: 13px;
            color: #334155;
            max-width: 650px;
        }}
        .action-tag {{
            background: {accent_hex};
            color: #ffffff;
            font-weight: 700;
            font-size: 12px;
            padding: 6px 14px;
            border-radius: 20px;
            letter-spacing: 0.5px;
        }}
        .images-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 28px;
        }}
        .image-card {{
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            text-align: center;
        }}
        .image-card img {{
            width: 100%;
            height: 260px;
            object-fit: cover;
            display: block;
        }}
        .image-label {{
            background: #f8fafc;
            color: #334155;
            padding: 8px 12px;
            font-size: 12px;
            font-weight: 600;
            border-top: 1px solid #e2e8f0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 28px;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
            text-align: left;
        }}
        th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        .scale-badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
        }}
        .scale-small {{ background: #e0f2fe; color: #0369a1; }}
        .scale-medium {{ background: #fef3c7; color: #b45309; }}
        .scale-large {{ background: #fee2e2; color: #b91c1c; }}
        .footer {{
            border-top: 1px solid #e2e8f0;
            padding-top: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: #94a3b8;
        }}
        @media print {{
            body {{ padding: 0; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="brand">
            <h1>CIVIL ROAD DEFECT & DAMAGE AUDIT REPORT</h1>
            <p>Automated Multi-Scale Computer Vision Inspection System | {model_used}</p>
        </div>
        <div class="meta-box">
            <div><strong>Audit Reference:</strong> #{rec_id}</div>
            <div><strong>Timestamp:</strong> {ts}</div>
            <div><strong>File Source:</strong> {filename}</div>
        </div>
    </div>

    <div class="severity-banner">
        <div>
            <div class="severity-title">STATUS: {level.upper()} RISK (DSI {severity}/100)</div>
            <div class="severity-desc">{rec}</div>
        </div>
        <div class="action-tag">{action_code}</div>
    </div>

    <div class="summary-grid">
        <div class="stat-card">
            <div class="stat-label">Pothole Defect Count</div>
            <div class="stat-val">{pothole_count}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Surface Damage Ratio</div>
            <div class="stat-val">{damage_pct}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Scale Distribution</div>
            <div class="stat-val" style="font-size: 18px; line-height: 32px;">
                <span style="color:#0284c7;">{small_c}S</span> &bull; 
                <span style="color:#d97706;">{med_c}M</span> &bull; 
                <span style="color:#dc2626;">{large_c}L</span>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Severity Index</div>
            <div class="stat-val" style="color: {accent_hex};">{severity} <span style="font-size:14px;color:#64748b;">/ 100</span></div>
        </div>
    </div>

    {"<div class='images-section'><div class='image-card'><img src='" + original_base64 + "' alt='Original'><div class='image-label'>Original Surface Capture</div></div><div class='image-card'><img src='" + overlay_base64 + "' alt='Overlay'><div class='image-label'>MultiScale-RoadNet Segmentation & Bounding Overlay</div></div></div>" if original_base64 and overlay_base64 else ""}

    <h3 style="font-size: 15px; color:#0f172a; margin-bottom: 12px;">Localized Damage Inventory</h3>
    <table>
        <thead>
            <tr>
                <th>Defect ID</th>
                <th>Bounding BBox [X, Y, W, H]</th>
                <th>Area (Pixels)</th>
                <th>Scale Classification</th>
                <th>Confidence</th>
                <th>Depth Proxy</th>
            </tr>
        </thead>
        <tbody>
            {instance_rows}
        </tbody>
    </table>

    <div class="footer">
        <div>Generated by MultiScale-RoadNet Inspection Engine (ResNet-ASPP-FPN)</div>
        <div>Standard: ASTM D6433 Pavement Condition Index (PCI) Compatible</div>
    </div>
</body>
</html>
    """
    return html
