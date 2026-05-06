import threading
import queue
import logging
import re
from datetime import datetime, timedelta
try:
    import pyodbc
except ImportError:
    pyodbc = None

from config import DB_CONFIG, build_db_connection_string

log = logging.getLogger(__name__)


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


def _quoted_identifier(value: str) -> str:
    return f"[{value}]"


def _qualified_table(table_key: str) -> str:
    schema = _safe_identifier(DB_CONFIG.get("schema"), "schema")
    table = _safe_identifier(DB_CONFIG.get(table_key), table_key)
    return f"{_quoted_identifier(schema)}.{_quoted_identifier(table)}"


PERSONS_TABLE_SQL = _qualified_table("persons_table")
ATTENDANCE_TABLE_SQL = _qualified_table("attendance_table")


def _ensure_pyodbc():
    if pyodbc is None:
        raise RuntimeError(
            "pyodbc is not installed in the active interpreter. "
            "Activate the project virtual environment or install requirements.txt."
        )

# ============================================================
# CONNECTION
# ============================================================
def get_connection():
    _ensure_pyodbc()
    conn_str = build_db_connection_string()
    return pyodbc.connect(conn_str, timeout=DB_CONFIG["timeout"])


def test_connection():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        missing = []
        for label, table_name in (
            ("persons_table", PERSONS_TABLE_SQL),
            ("attendance_table", ATTENDANCE_TABLE_SQL),
        ):
            cursor.execute("SELECT OBJECT_ID(?, 'U')", (table_name,))
            if cursor.fetchone()[0] is None:
                missing.append(f"{label}={table_name}")
        conn.close()
        if missing:
            log.error(
                "[ERR] Database connected, but required tables were not found in %s: %s",
                DB_CONFIG["database"],
                ", ".join(missing),
            )
            return False
        log.info(
            "[OK] Database connection successful. Using %s and %s",
            PERSONS_TABLE_SQL,
            ATTENDANCE_TABLE_SQL,
        )
        return True
    except ValueError as e:
        log.error(f"[ERR] Invalid database configuration: {e}")
        return False
    except Exception as e:
        msg = str(e)
        if "Encryption not supported on the client" in msg:
            log.error(
                "[ERR] Database connection FAILED: %s | Check SQL Server TLS/encryption settings "
                "or set DB_SERVER/DB_ENCRYPT/DB_TRUST_SERVER_CERTIFICATE to match the instance.",
                e,
            )
        else:
            log.error(f"[ERR] Database connection FAILED: {e}")
        return False


# ============================================================
# CORE OPERATIONS
# ============================================================
def _do_log_arrival(name: str):
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Step 1: Get person_id
        cursor.execute(f"SELECT person_id FROM {PERSONS_TABLE_SQL} WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            log.warning(f"'{name}' not found in {PERSONS_TABLE_SQL}. "
                        f"Check exact spelling — DB has: sasi, Sasidhar")
            return
        person_id = row[0]

        # Step 2: Already marked INSIDE?
        cursor.execute(
            f"SELECT 1 FROM {ATTENDANCE_TABLE_SQL} WHERE person_id = ? AND status = 'INSIDE'",
            (person_id,)
        )
        if cursor.fetchone():
            log.debug(f"'{name}' already INSIDE — skipping.")
            return

        # Step 3: 1-hour minimum gap between entries
        one_hour_ago = datetime.now() - timedelta(hours=1)
        cursor.execute(
            f"SELECT TOP 1 arrival_time FROM {ATTENDANCE_TABLE_SQL} "
            "WHERE person_id = ? ORDER BY arrival_time DESC",
            (person_id,)
        )
        last = cursor.fetchone()
        if last and last[0] > one_hour_ago:
            mins = int((datetime.now() - last[0]).total_seconds() / 60)
            log.info(f"'{name}' was logged {mins} min ago — skipping (need 60 min gap).")
            return

        # Step 4: Insert
        cursor.execute(
            f"INSERT INTO {ATTENDANCE_TABLE_SQL} (person_id, arrival_time, status) VALUES (?, ?, 'INSIDE')",
            (person_id, datetime.now())
        )
        conn.commit()
        log.info(f"[OK] Arrival logged → {name} (person_id={person_id})")

    except Exception as e:
        msg = str(e)
        if "Encryption not supported on the client" in msg:
            log.error(
                "[ERR] SQL connection error in log_arrival('%s'): %s | Check SQL Server encryption/TLS "
                "settings and DB_SERVER=%s",
                name,
                e,
                DB_CONFIG["server"],
            )
        elif "Invalid object name" in msg:
            log.error(
                "[ERR] SQL error in log_arrival('%s'): %s | Check DB_NAME=%s, DB_SCHEMA=%s, "
                "DB_PERSONS_TABLE=%s, DB_ATTENDANCE_TABLE=%s",
                name,
                e,
                DB_CONFIG["database"],
                DB_CONFIG["schema"],
                DB_CONFIG["persons_table"],
                DB_CONFIG["attendance_table"],
            )
        else:
            log.error(f"[ERR] Error in log_arrival('{name}'): {e}")
    finally:
        if conn:
            try: conn.close()
            except: pass


def _do_log_departure(name: str):
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"SELECT person_id FROM {PERSONS_TABLE_SQL} WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return
        person_id = row[0]

        cursor.execute(
            f"UPDATE {ATTENDANCE_TABLE_SQL} SET departure_time = ?, status = 'LEFT' "
            "WHERE person_id = ? AND status = 'INSIDE'",
            (datetime.now(), person_id)
        )
        if cursor.rowcount > 0:
            conn.commit()
            log.info(f"[OK] Departure logged → {name}")

    except Exception as e:
        msg = str(e)
        if "Encryption not supported on the client" in msg:
            log.error(
                "[ERR] SQL connection error in log_departure('%s'): %s | Check SQL Server encryption/TLS "
                "settings and DB_SERVER=%s",
                name,
                e,
                DB_CONFIG["server"],
            )
        elif "Invalid object name" in msg:
            log.error(
                "[ERR] SQL error in log_departure('%s'): %s | Check DB_NAME=%s, DB_SCHEMA=%s, "
                "DB_PERSONS_TABLE=%s, DB_ATTENDANCE_TABLE=%s",
                name,
                e,
                DB_CONFIG["database"],
                DB_CONFIG["schema"],
                DB_CONFIG["persons_table"],
                DB_CONFIG["attendance_table"],
            )
        else:
            log.error(f"[ERR] Error in log_departure('{name}'): {e}")
    finally:
        if conn:
            try: conn.close()
            except: pass


def close_all_inside_attendance(shutdown_time: datetime | None = None) -> int:
    """Mark every currently INSIDE attendance row as LEFT at shutdown time."""
    conn = None
    when = shutdown_time or datetime.now()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {ATTENDANCE_TABLE_SQL} "
            "SET departure_time = ?, status = 'LEFT' "
            "WHERE status = 'INSIDE'",
            (when,),
        )
        affected = max(cursor.rowcount, 0)
        conn.commit()
        if affected:
            log.info(
                "[OK] Shutdown attendance closeout complete. Marked %d row(s) LEFT at %s",
                affected,
                when.isoformat(timespec="seconds"),
            )
        else:
            log.info(
                "[OK] Shutdown attendance closeout complete. No INSIDE rows to update at %s",
                when.isoformat(timespec="seconds"),
            )
        return affected
    except Exception as e:
        msg = str(e)
        if "Encryption not supported on the client" in msg:
            log.error(
                "[ERR] SQL connection error during shutdown attendance closeout: %s | "
                "Check SQL Server encryption/TLS settings and DB_SERVER=%s",
                e,
                DB_CONFIG["server"],
            )
        elif "Invalid object name" in msg:
            log.error(
                "[ERR] SQL error during shutdown attendance closeout: %s | "
                "Check DB_NAME=%s, DB_SCHEMA=%s, DB_PERSONS_TABLE=%s, DB_ATTENDANCE_TABLE=%s",
                e,
                DB_CONFIG["database"],
                DB_CONFIG["schema"],
                DB_CONFIG["persons_table"],
                DB_CONFIG["attendance_table"],
            )
        else:
            log.error(f"[ERR] Error during shutdown attendance closeout: {e}")
        return 0
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ============================================================
# ASYNC QUEUE  (DB writes never block the camera thread)
# ============================================================
_db_queue: queue.Queue = queue.Queue()

def _db_worker():
    while True:
        try:
            func, args = _db_queue.get(timeout=1)
            func(*args)
            _db_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            log.error(f"[ERR] DB worker error: {e}")

threading.Thread(target=_db_worker, daemon=True).start()

# ============================================================
# PUBLIC API
# ============================================================
def log_arrival(name: str):
    _db_queue.put((_do_log_arrival, (name,)))

def log_departure(name: str):
    _db_queue.put((_do_log_departure, (name,)))

def flush_db_queue():
    """Block until all pending DB writes complete. Call on exit."""
    try:
        _db_queue.join()
        log.info("[OK] All DB writes completed.")
    except Exception as e:
        log.error(f"[ERR] DB flush error: {e}")
