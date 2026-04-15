import logging
import os
import sys
import hashlib
from flask import Flask, Response, jsonify, render_template
import cv2
import time
import pyodbc
import numpy as np
from datetime import datetime, date
from config import DB_CONFIG

app = Flask(__name__)
log = logging.getLogger("flask_app")


def _runtime_module():
    """Return the live security system module, even when run as __main__."""
    main_mod = sys.modules.get("__main__")
    if main_mod and getattr(main_mod, "__file__", "").endswith("security_system.py"):
        return main_mod
    return sys.modules.get("security_system")


def _runtime_attr(name, default=None):
    module = _runtime_module()
    return getattr(module, name, default) if module is not None else default


def _db_ready() -> bool:
    return bool(DB_CONFIG.get("server") and DB_CONFIG.get("database"))


def _placeholder_frame(message: str) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (8, 14, 24)
    cv2.putText(
        frame,
        "AI SECURITY DASHBOARD",
        (70, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 229, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (70, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200, 220, 232),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Start security_system.py to publish live frames.",
        (70, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (74, 122, 153),
        2,
        cv2.LINE_AA,
    )
    return frame

# ============================================================
# VIDEO STREAM
# ============================================================
def gen_frames():
    """Generator that yields MJPEG frames from security_system's latest_frame."""
    placeholder = _placeholder_frame("Waiting for camera stream...")
    while True:
        try:
            latest_frame = _runtime_attr("latest_frame")
            frame = latest_frame if latest_frame is not None else placeholder
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        except Exception:
            log.exception("Video feed generation failed.")
        time.sleep(0.04)  # ~25 fps


@app.route('/video_feed')
def video_feed():
    response = Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


# ============================================================
# DB HELPER
# ============================================================
def get_db():
    if not _db_ready():
        raise RuntimeError(
            "Database is not configured. Set DB_SERVER and DB_NAME environment variables."
        )
    conn_str = (
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str, timeout=5)


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logs')
def get_logs():
    """Return today's attendance + intruder events for the dashboard log panel."""
    try:
        today = date.today()
        events = []

        if _db_ready():
            conn  = get_db()
            cur   = conn.cursor()

            # Authorized arrivals/departures from Attendance
            cur.execute("""
                SELECT p.name,
                       a.arrival_time,
                       a.departure_time,
                       a.status
                FROM   Attendance a
                JOIN   Persons p ON p.person_id = a.person_id
                WHERE  CAST(a.arrival_time AS DATE) = ?
                ORDER  BY a.arrival_time DESC
            """, (today,))

            rows = cur.fetchall()

            for name, arr, dep, status in rows:
                events.append({
                    "name"     : name,
                    "event"    : "arrival",
                    "timestamp": str(arr)[:19] if arr else "",
                    "status"   : status
                })
                if dep:
                    events.append({
                        "name"     : name,
                        "event"    : "departure",
                        "timestamp": str(dep)[:19],
                        "status"   : "LEFT"
                    })

            conn.close()

        # Intruder events from security_system memory
        try:
            intruder_db = _runtime_attr("intruder_db")
            import json
            if intruder_db is not None and os.path.exists(intruder_db.path):
                with open(intruder_db.path) as f:
                    db = json.load(f)
                for iid, rec in db.get("records", {}).items():
                    first = rec.get("first_seen", "")
                    if first[:10] == str(today):
                        events.append({
                            "name"     : iid,
                            "event"    : "intruder",
                            "timestamp": first[:19],
                            "status"   : "INTRUDER"
                        })
        except Exception:
            pass

        # Sort all events newest first
        events.sort(key=lambda x: x["timestamp"], reverse=True)
        return jsonify(events)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alerts')
def get_alerts():
    """Return the latest alert from security_system."""
    try:
        latest_alert = _runtime_attr("latest_alert")
        if not latest_alert:
            return jsonify({})

        is_intruder = "intruder" in latest_alert.lower() or "INTRUDER" in latest_alert
        alert_id = hashlib.sha1(latest_alert.encode("utf-8")).hexdigest()[:16]
        return jsonify({
            "alert"    : latest_alert,
            "type"     : "intruder" if is_intruder else "authorized",
            "name"     : latest_alert,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id"       : alert_id,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats')
def get_stats():
    """Return live counts for dashboard stat cards."""
    try:
        today = date.today()
        present = 0
        authorized = 0
        total = 0

        if _db_ready():
            conn = get_db()
            cur  = conn.cursor()

            # Currently INSIDE
            cur.execute("""
                SELECT COUNT(*) FROM Attendance
                WHERE status = 'INSIDE'
                AND CAST(arrival_time AS DATE) = ?
            """, (today,))
            present = cur.fetchone()[0]

            # Unique authorized today
            cur.execute("""
                SELECT COUNT(DISTINCT person_id) FROM Attendance
                WHERE CAST(arrival_time AS DATE) = ?
            """, (today,))
            authorized = cur.fetchone()[0]

            # Total events today
            cur.execute("""
                SELECT COUNT(*) FROM Attendance
                WHERE CAST(arrival_time AS DATE) = ?
            """, (today,))
            total = cur.fetchone()[0]

            conn.close()

        # Intruder count from JSON
        intruders = 0
        try:
            intruder_db = _runtime_attr("intruder_db")
            import json
            if intruder_db is not None and os.path.exists(intruder_db.path):
                with open(intruder_db.path) as f:
                    db = json.load(f)
                for rec in db.get("records", {}).values():
                    if rec.get("first_seen", "")[:10] == str(today):
                        intruders += 1
        except Exception:
            pass

        return jsonify({
            "present"   : present,
            "authorized": authorized,
            "intruders" : intruders,
            "total"     : total
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/attendance')
def get_attendance():
    """Return full attendance table for the logs modal."""
    try:
        if not _db_ready():
            return jsonify([])

        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT p.name,
                   a.arrival_time,
                   a.departure_time,
                   a.status
            FROM   Attendance a
            JOIN   Persons p ON p.person_id = a.person_id
            ORDER  BY a.arrival_time DESC
        """)
        rows = cur.fetchall()
        conn.close()
        return jsonify([
            {
                "name"          : r[0],
                "arrival_time"  : str(r[1])[:19] if r[1] else "",
                "departure_time": str(r[2])[:19] if r[2] else "—",
                "status"        : r[3]
            }
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
