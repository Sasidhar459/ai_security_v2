import pyodbc
import threading
import queue
import logging
from datetime import datetime, timedelta
from config import DB_CONFIG

log = logging.getLogger(__name__)

# ============================================================
# CONNECTION
# ============================================================
def get_connection():
    conn_str = (
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str, timeout=5)


def test_connection():
    try:
        conn = get_connection()
        conn.close()
        log.info("[OK] Database connection successful.")
        return True
    except pyodbc.Error as e:
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
        cursor.execute("SELECT person_id FROM Persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            log.warning(f"'{name}' not found in Persons table. "
                        f"Check exact spelling — DB has: sasi, Sasidhar")
            return
        person_id = row[0]

        # Step 2: Already marked INSIDE?
        cursor.execute(
            "SELECT 1 FROM Attendance WHERE person_id = ? AND status = 'INSIDE'",
            (person_id,)
        )
        if cursor.fetchone():
            log.debug(f"'{name}' already INSIDE — skipping.")
            return

        # Step 3: 1-hour minimum gap between entries
        one_hour_ago = datetime.now() - timedelta(hours=1)
        cursor.execute(
            "SELECT TOP 1 arrival_time FROM Attendance "
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
            "INSERT INTO Attendance (person_id, arrival_time, status) VALUES (?, ?, 'INSIDE')",
            (person_id, datetime.now())
        )
        conn.commit()
        log.info(f"[OK] Arrival logged → {name} (person_id={person_id})")

    except pyodbc.Error as e:
        log.error(f"[ERR] SQL error in log_arrival('{name}'): {e}")
    except Exception as e:
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

        cursor.execute("SELECT person_id FROM Persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return
        person_id = row[0]

        cursor.execute(
            "UPDATE Attendance SET departure_time = ?, status = 'LEFT' "
            "WHERE person_id = ? AND status = 'INSIDE'",
            (datetime.now(), person_id)
        )
        if cursor.rowcount > 0:
            conn.commit()
            log.info(f"[OK] Departure logged → {name}")

    except pyodbc.Error as e:
        log.error(f"[ERR] SQL error in log_departure('{name}'): {e}")
    except Exception as e:
        log.error(f"[ERR] Error in log_departure('{name}'): {e}")
    finally:
        if conn:
            try: conn.close()
            except: pass


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