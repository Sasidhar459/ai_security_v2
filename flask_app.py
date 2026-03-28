from flask import Flask, Response, jsonify, render_template
import cv2
import time
import pyodbc
from datetime import datetime, date
from config import DB_CONFIG

app = Flask(__name__)

# ============================================================
# VIDEO STREAM
# ============================================================
def gen_frames():
    """Generator that yields MJPEG frames from security_system's latest_frame."""
    while True:
        try:
            from security_system import latest_frame
            if latest_frame is not None:
                ret, buf = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buf.tobytes() + b'\r\n')
        except Exception:
            pass
        time.sleep(0.04)  # ~25 fps


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# DB HELPER
# ============================================================
def get_db():
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
        conn  = get_db()
        cur   = conn.cursor()
        today = date.today()

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

        rows   = cur.fetchall()
        events = []

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

        # Intruder events from security_system memory
        try:
            from security_system import intruder_db
            import json, os
            if os.path.exists(intruder_db.path):
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

        conn.close()

        # Sort all events newest first
        events.sort(key=lambda x: x["timestamp"], reverse=True)
        return jsonify(events)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alerts')
def get_alerts():
    """Return the latest alert from security_system."""
    try:
        from security_system import latest_alert
        if not latest_alert:
            return jsonify({})

        is_intruder = "intruder" in latest_alert.lower() or "INTRUDER" in latest_alert
        return jsonify({
            "alert"    : latest_alert,
            "type"     : "intruder" if is_intruder else "authorized",
            "name"     : latest_alert,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id"       : hash(latest_alert + str(int(time.time() / 10)))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats')
def get_stats():
    """Return live counts for dashboard stat cards."""
    try:
        conn = get_db()
        cur  = conn.cursor()
        today = date.today()

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
            from security_system import intruder_db
            import json, os
            if os.path.exists(intruder_db.path):
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