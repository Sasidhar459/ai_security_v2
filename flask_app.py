import logging
import os
import sys
import hashlib
import re
from contextlib import contextmanager
from flask import Flask, Response, jsonify, render_template
import cv2
import time
try:
    import pyodbc
except ImportError:
    pyodbc = None
import numpy as np
from datetime import datetime, date
from config import DB_CONFIG, build_db_connection_string

app = Flask(__name__)
log = logging.getLogger("flask_app")
_DB_STATUS_CACHE = {
    "checked_at": 0.0,
    "status": {
        "configured": False,
        "connected": False,
        "message": "Database status not checked yet.",
    },
}


def _safe_identifier(value: str, label: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"DB {label} is empty.")
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(
            f"DB {label} contains unsupported characters: {value!r}. "
            "Use letters, numbers, or underscore only."
        )
    return value


def _qualified_table(table_key: str) -> str:
    schema = _safe_identifier(DB_CONFIG.get("schema"), "schema")
    table = _safe_identifier(DB_CONFIG.get(table_key), table_key)
    return f"[{schema}].[{table}]"


PERSONS_TABLE_SQL = _qualified_table("persons_table")
ATTENDANCE_TABLE_SQL = _qualified_table("attendance_table")


def _ensure_pyodbc():
    if pyodbc is None:
        raise RuntimeError(
            "pyodbc is not installed in the active interpreter. "
            "Activate the project virtual environment or install requirements.txt."
        )


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


def _format_timestamp(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    return text[:19] if text else ""


def _parse_iso_datetime(value):
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _runtime_state_snapshot() -> dict:
    reader = _runtime_attr("read_runtime_state")
    if callable(reader):
        try:
            data = reader()
            return data if isinstance(data, dict) else {}
        except Exception:
            log.exception("Failed to read runtime state from security_system.")
    return {}


def _runtime_timestamp(attr_name: str):
    return _parse_iso_datetime(_runtime_attr(attr_name))


def _intruder_records() -> list[tuple[str, dict]]:
    intruder_db = _runtime_attr("intruder_db")
    if intruder_db is None or not os.path.exists(intruder_db.path):
        return []
    try:
        import json
        with open(intruder_db.path, encoding="utf-8") as f:
            db = json.load(f)
        return list(db.get("records", {}).items())
    except Exception:
        log.exception("Failed to read intruder database for dashboard.")
        return []


@contextmanager
def _db_cursor():
    conn = None
    try:
        conn = get_db()
        yield conn.cursor()
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _db_health_snapshot(ttl_secs: float = 10.0) -> dict:
    now = time.time()
    if now - _DB_STATUS_CACHE["checked_at"] < ttl_secs:
        return dict(_DB_STATUS_CACHE["status"])

    status = {
        "configured": _db_ready(),
        "connected": False,
        "message": "Database not configured.",
    }
    if status["configured"]:
        try:
            with _db_cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            status["connected"] = True
            status["message"] = "Database connected."
        except Exception as e:
            status["message"] = str(e)

    _DB_STATUS_CACHE["checked_at"] = now
    _DB_STATUS_CACHE["status"] = status
    return dict(status)


def _system_status_payload() -> dict:
    runtime_state = _runtime_state_snapshot()
    frame_time = _runtime_timestamp("latest_frame_updated_at")
    alert_time = _runtime_timestamp("latest_alert_updated_at")
    now = datetime.now()

    stream_age = None
    stream_live = False
    if frame_time is not None:
        stream_age = max((now - frame_time).total_seconds(), 0.0)
        stream_live = stream_age <= 5.0

    latest_alert = _runtime_attr("latest_alert")
    db_status = _db_health_snapshot()
    runtime_reason = runtime_state.get("reason") or (
        "running" if runtime_state.get("active") else "idle"
    )

    return {
        "system_active": bool(runtime_state.get("active")),
        "runtime_reason": runtime_reason,
        "runtime_updated_at": _format_timestamp(runtime_state.get("updated_at")),
        "last_heartbeat": _format_timestamp(runtime_state.get("last_heartbeat")),
        "stream_live": stream_live,
        "stream_age_seconds": None if stream_age is None else round(stream_age, 1),
        "last_frame_at": _format_timestamp(frame_time),
        "has_alert": bool(latest_alert),
        "last_alert_at": _format_timestamp(alert_time),
        "db_configured": db_status["configured"],
        "db_connected": db_status["connected"],
        "db_message": db_status["message"],
        "test_people": _runtime_attr("REALTIME_TRUE_PEOPLE", "?"),
    }


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
    _ensure_pyodbc()
    if not _db_ready():
        raise RuntimeError(
            "Database is not configured. Set DB_SERVER and DB_NAME environment variables."
        )
    conn_str = build_db_connection_string()
    return pyodbc.connect(conn_str, timeout=DB_CONFIG["timeout"])


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/system_status')
def get_system_status():
    return jsonify(_system_status_payload())


@app.route('/logs')
def get_logs():
    """Return today's attendance + intruder events for the dashboard log panel."""
    today = date.today()
    events = []

    if _db_ready():
        try:
            with _db_cursor() as cur:
                cur.execute("""
                    SELECT p.name,
                           a.arrival_time,
                           a.departure_time,
                           a.status
                    FROM   {attendance} a
                    JOIN   {persons} p ON p.person_id = a.person_id
                    WHERE  CAST(a.arrival_time AS DATE) = ?
                    ORDER  BY a.arrival_time DESC
                """.format(attendance=ATTENDANCE_TABLE_SQL, persons=PERSONS_TABLE_SQL), (today,))

                for name, arr, dep, status in cur.fetchall():
                    events.append({
                        "name": name,
                        "event": "arrival",
                        "timestamp": _format_timestamp(arr),
                        "status": status,
                    })
                    if dep:
                        events.append({
                            "name": name,
                            "event": "departure",
                            "timestamp": _format_timestamp(dep),
                            "status": "LEFT",
                        })
        except Exception:
            log.exception("Dashboard /logs failed to read attendance data.")

    for iid, rec in _intruder_records():
        first = _format_timestamp(rec.get("first_seen"))
        if first[:10] == str(today):
            events.append({
                "name": iid,
                "event": "intruder",
                "timestamp": first,
                "status": "INTRUDER",
            })

    events.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(events)


@app.route('/alerts')
def get_alerts():
    """Return the latest alert from security_system."""
    try:
        latest_alert = _runtime_attr("latest_alert")
        if not latest_alert:
            return jsonify({})

        if isinstance(latest_alert, dict):
            alert_text = (
                latest_alert.get("text")
                or latest_alert.get("message")
                or latest_alert.get("title")
                or ""
            )
            alert_type = latest_alert.get("type") or (
                "intruder" if "intruder" in alert_text.lower() else "authorized"
            )
            alert_id = latest_alert.get("id") or hashlib.sha1(
                alert_text.encode("utf-8")
            ).hexdigest()[:16]
            return jsonify({
                "id": alert_id,
                "type": alert_type,
                "severity": latest_alert.get("severity", "info"),
                "title": latest_alert.get("title", "System Alert"),
                "name": latest_alert.get("name") or alert_text or "Alert",
                "message": latest_alert.get("message") or alert_text,
                "alert": alert_text,
                "timestamp": latest_alert.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        alert_text = str(latest_alert)
        is_intruder = "intruder" in alert_text.lower() or "INTRUDER" in alert_text
        alert_id = hashlib.sha1(alert_text.encode("utf-8")).hexdigest()[:16]
        return jsonify({
            "alert": alert_text,
            "message": alert_text,
            "title": "Intruder Detected" if is_intruder else "Authorized Access",
            "type": "intruder" if is_intruder else "authorized",
            "name": alert_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": alert_id,
        })
    except Exception as e:
        log.exception("Dashboard /alerts failed.")
        return jsonify({})


@app.route('/stats')
def get_stats():
    """Return live counts for dashboard stat cards."""
    today = date.today()
    present = 0
    authorized = 0
    total = 0

    if _db_ready():
        try:
            with _db_cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM {attendance}
                    WHERE status = 'INSIDE'
                    AND CAST(arrival_time AS DATE) = ?
                """.format(attendance=ATTENDANCE_TABLE_SQL), (today,))
                present = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(DISTINCT person_id) FROM {attendance}
                    WHERE CAST(arrival_time AS DATE) = ?
                """.format(attendance=ATTENDANCE_TABLE_SQL), (today,))
                authorized = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*) FROM {attendance}
                    WHERE CAST(arrival_time AS DATE) = ?
                """.format(attendance=ATTENDANCE_TABLE_SQL), (today,))
                total = cur.fetchone()[0]
        except Exception:
            log.exception("Dashboard /stats failed to read attendance data.")

    intruders = 0
    for _, rec in _intruder_records():
        if _format_timestamp(rec.get("first_seen"))[:10] == str(today):
            intruders += 1

    system_status = _system_status_payload()
    return jsonify({
        "present": present,
        "authorized": authorized,
        "intruders": intruders,
        "total": total,
        "db_connected": system_status["db_connected"],
        "stream_live": system_status["stream_live"],
    })


@app.route('/attendance')
def get_attendance():
    """Return full attendance table for the logs modal."""
    if not _db_ready():
        return jsonify([])

    try:
        with _db_cursor() as cur:
            cur.execute("""
                SELECT p.name,
                       a.arrival_time,
                       a.departure_time,
                       a.status
                FROM   {attendance} a
                JOIN   {persons} p ON p.person_id = a.person_id
                ORDER  BY a.arrival_time DESC
            """.format(attendance=ATTENDANCE_TABLE_SQL, persons=PERSONS_TABLE_SQL))
            return jsonify([
                {
                    "name": r[0],
                    "arrival_time": _format_timestamp(r[1]),
                    "departure_time": _format_timestamp(r[2]) or "—",
                    "status": r[3],
                }
                for r in cur.fetchall()
            ])
    except Exception:
        log.exception("Dashboard /attendance failed.")
        return jsonify([])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
