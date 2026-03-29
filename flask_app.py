from flask import Flask, Response, jsonify, render_template
import cv2
import time
import hashlib
import json
import os
import logging
import pyodbc
from datetime import datetime, date
from config import DB_CONFIG

import shared_state  # holds latest_frame, latest_alert, intruder_db

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DB CONNECTION
# ============================================================
def get_db():
    conn_str = (
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str, timeout=5)


def stable_id(value: str) -> str:
    """Return a stable hash ID safe across processes."""
    return hashlib.md5(value.encode()).hexdigest()[:12]


# ============================================================
# VIDEO STREAM
# ============================================================
def gen_frames():
    """Generator that yields MJPEG frames from shared_state.latest_frame."""
    while True:
        frame = shared_state.latest_frame
        if frame is not None:
            try:
                ret, buf = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if ret:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + buf.tobytes()
                        + b'\r\n'
                    )
            except Exception:
                logger.exception("Error encoding video frame")
        time.sleep(0.04)  # ~25 fps


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ============================================================
# HELPERS
# ============================================================
def get_todays_intruders(today: date) -> list:
    """Read intruder records for today from the JSON store."""
    intruders = []
    try:
        db = shared_state.intruder_db
        if db is None:
            return intruders
        path = db.path
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for iid, rec in data.get("records", {}).items():
                first = rec.get("first_seen", "")
                if first[:10] == str(today):
                    intruders.append({
                        "name"          : iid,
                        "event"         : "intruder",
                        "timestamp"     : first[:19],
                        "status"        : "INTRUDER",
                        "alerts_stopped": rec.get("alerts_stopped", False),
                    })
    except Exception:
        logger.exception("Failed to read intruder DB")
    return intruders


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
        conn  = get_db()
        cur   = conn.cursor()
        today = date.today()

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

        events = []
        for name, arr, dep, status in cur.fetchall():
            if arr:
                events.append({
                    "name"     : name,
                    "event"    : "arrival",
                    "timestamp": str(arr)[:19],
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
        events.extend(get_todays_intruders(today))
        events.sort(key=lambda x: x["timestamp"], reverse=True)
        return jsonify(events)

    except Exception:
        logger.exception("Error in /logs")
        return jsonify({"error": "Failed to fetch logs"}), 500


@app.route('/alerts')
def get_alerts():
    """Return the latest alert from shared_state."""
    try:
        alert = shared_state.latest_alert
        if not alert:
            return jsonify({})

        is_intruder = "intruder" in alert.lower() or "INTRUDER" in alert
        return jsonify({
            "alert"    : alert,
            "type"     : "intruder" if is_intruder else "authorized",
            "name"     : alert,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id"       : stable_id(alert + str(int(time.time() / 10)))
        })
    except Exception:
        logger.exception("Error in /alerts")
        return jsonify({"error": "Failed to fetch alerts"}), 500


@app.route('/stats')
def get_stats():
    """Return live counts for dashboard stat cards."""
    try:
        conn  = get_db()
        cur   = conn.cursor()
        today = date.today()

        cur.execute("""
            SELECT
                SUM(CASE WHEN status = 'INSIDE' THEN 1 ELSE 0 END),
                COUNT(DISTINCT person_id),
                COUNT(*)
            FROM Attendance
            WHERE CAST(arrival_time AS DATE) = ?
        """, (today,))

        present, authorized, total = cur.fetchone()
        conn.close()

        intruders = len(get_todays_intruders(today))

        return jsonify({
            "present"   : present    or 0,
            "authorized": authorized or 0,
            "intruders" : intruders,
            "total"     : total      or 0
        })

    except Exception:
        logger.exception("Error in /stats")
        return jsonify({"error": "Failed to fetch stats"}), 500


@app.route('/attendance')
def get_attendance():
    """Return full attendance table for the logs modal."""
    try:
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
                "departure_time": str(r[2])[:19] if r[2] else "",
                "status"        : r[3]
            }
            for r in rows
        ])
    except Exception:
        logger.exception("Error in /attendance")
        return jsonify({"error": "Failed to fetch attendance"}), 500


# ============================================================
# INTRUDER ALERT CONTROL
# ============================================================
@app.route('/stop_alert/<iid>', methods=['POST'])
def stop_alert(iid):
    """Stop further Telegram alerts for a specific intruder (user dismissed)."""
    try:
        db = shared_state.intruder_db
        if db is None:
            return jsonify({"error": "IntruderDB not ready"}), 503
        db.stop_alerts(iid)
        shared_state.latest_alert = ""   # clear dashboard banner too
        return jsonify({"status": "stopped", "iid": iid})
    except Exception:
        logger.exception("Error in /stop_alert/%s", iid)
        return jsonify({"error": "Failed to stop alert"}), 500


@app.route('/resume_alert/<iid>', methods=['POST'])
def resume_alert(iid):
    """Re-enable Telegram alerts for a specific intruder."""
    try:
        db = shared_state.intruder_db
        if db is None:
            return jsonify({"error": "IntruderDB not ready"}), 503
        db.resume_alerts(iid)
        return jsonify({"status": "resumed", "iid": iid})
    except Exception:
        logger.exception("Error in /resume_alert/%s", iid)
        return jsonify({"error": "Failed to resume alert"}), 500


@app.route('/intruders')
def get_intruders():
    """Return all known intruder IDs and their alert-stopped status for the dashboard."""
    try:
        db = shared_state.intruder_db
        if db is None:
            return jsonify([])
        today = str(date.today())
        result = []
        with db.lock:
            for iid, rec in db._data.get("records", {}).items():
                if rec.get("first_seen", "")[:10] == today:
                    result.append({
                        "iid"           : iid,
                        "first_seen"    : rec.get("first_seen", "")[:19],
                        "alert_count"   : rec.get("alert_count", 0),
                        "alerts_stopped": rec.get("alerts_stopped", False),
                    })
        return jsonify(result)
    except Exception:
        logger.exception("Error in /intruders")
        return jsonify({"error": "Failed to fetch intruders"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)