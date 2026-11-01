"""
SQLite Database Layer for Road Damage Inspection History.
Persists inspection records, timestamps, severity metrics, and base64 thumbnails.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).resolve().parent / "inspections.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            thumbnail_base64 TEXT,
            model_used TEXT,
            severity_score INTEGER,
            severity_level TEXT,
            action_code TEXT,
            badge_color TEXT,
            pothole_count INTEGER,
            damaged_area_percentage REAL,
            small_count INTEGER,
            medium_count INTEGER,
            large_count INTEGER,
            recommendation TEXT,
            instances_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_inspection(data: Dict[str, Any]) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO inspections (
            filename, thumbnail_base64, model_used,
            severity_score, severity_level, action_code, badge_color,
            pothole_count, damaged_area_percentage,
            small_count, medium_count, large_count,
            recommendation, instances_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("filename", "unknown.jpg"),
        data.get("thumbnail_base64", ""),
        data.get("model_used", "MultiScale-RoadNet"),
        data.get("severity_score", 0),
        data.get("severity_level", "None"),
        data.get("action_code", "SAFE"),
        data.get("badge_color", "emerald"),
        data.get("pothole_count", 0),
        data.get("damaged_area_percentage", 0.0),
        data.get("scale_distribution", {}).get("small", 0),
        data.get("scale_distribution", {}).get("medium", 0),
        data.get("scale_distribution", {}).get("large", 0),
        data.get("recommendation", ""),
        json.dumps(data.get("instances", [])),
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_all_inspections(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM inspections ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    results = []
    for row in rows:
        d = dict(row)
        if "instances_json" in d and d["instances_json"]:
            try:
                d["instances"] = json.loads(d["instances_json"])
            except Exception:
                d["instances"] = []
        else:
            d["instances"] = []
        results.append(d)
    conn.close()
    return results


def get_inspection_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inspections WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if "instances_json" in d and d["instances_json"]:
        try:
            d["instances"] = json.loads(d["instances_json"])
        except Exception:
            d["instances"] = []
    return d


def delete_inspection(record_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM inspections WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# Initialize on import
init_db()
