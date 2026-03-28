"""
Run this directly: python db_test.py
This bypasses the async queue and tests each step one by one.
"""
import pyodbc
from datetime import datetime, timedelta
from config import DB_CONFIG

def get_connection():
    conn_str = (
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str, timeout=5)

def run():
    print("\n========== DB DIAGNOSTIC ==========\n")

    # Step 1: Connection
    try:
        conn = get_connection()
        print("[PASS] Step 1: Connected to SQL Server")
    except Exception as e:
        print(f"[FAIL] Step 1: Cannot connect — {e}")
        return

    cursor = conn.cursor()

    # Step 2: List all persons
    print("\n[INFO] Step 2: All rows in Persons table:")
    try:
        cursor.execute("SELECT person_id, name FROM Persons")
        rows = cursor.fetchall()
        if not rows:
            print("  [WARN] Persons table is EMPTY!")
        for r in rows:
            print(f"  person_id={r[0]}  name='{r[1]}'  len={len(r[1])}")
    except Exception as e:
        print(f"  [FAIL] Cannot read Persons — {e}")
        conn.close()
        return

    # Step 3: Try lookup for each classifier name
    classifier_names = ["Sasidhar", "jyothsna"]  # from your log output
    print(f"\n[INFO] Step 3: Looking up classifier names: {classifier_names}")
    for name in classifier_names:
        cursor.execute("SELECT person_id FROM Persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            print(f"  [PASS] '{name}' -> person_id={row[0]}")
        else:
            print(f"  [FAIL] '{name}' NOT FOUND in Persons table!")
            print(f"         Check for trailing spaces or case mismatch.")
            # Try case-insensitive search
            cursor.execute("SELECT person_id, name FROM Persons WHERE LOWER(name) = LOWER(?)", (name,))
            row2 = cursor.fetchone()
            if row2:
                print(f"         Case-insensitive match found: '{row2[1]}' (person_id={row2[0]})")
                print(f"         Fix: UPDATE Persons SET name='{name}' WHERE person_id={row2[0]}")

    # Step 4: Check recent Attendance
    print("\n[INFO] Step 4: Last 5 rows in Attendance table:")
    try:
        cursor.execute("SELECT TOP 5 * FROM Attendance ORDER BY arrival_time DESC")
        rows = cursor.fetchall()
        if not rows:
            print("  Attendance table is EMPTY (no inserts have ever worked)")
        for r in rows:
            print(f"  {r}")
    except Exception as e:
        print(f"  [FAIL] Cannot read Attendance — {e}")

    # Step 5: Check 1-hour gap for Sasidhar
    print("\n[INFO] Step 5: 1-hour gap check for 'Sasidhar':")
    cursor.execute("SELECT person_id FROM Persons WHERE name = ?", ("Sasidhar",))
    row = cursor.fetchone()
    if row:
        person_id = row[0]
        one_hour_ago = datetime.now() - timedelta(hours=1)
        cursor.execute(
            "SELECT TOP 1 arrival_time FROM Attendance WHERE person_id = ? ORDER BY arrival_time DESC",
            (person_id,)
        )
        last = cursor.fetchone()
        if last:
            mins = int((datetime.now() - last[0]).total_seconds() / 60)
            if last[0] > one_hour_ago:
                print(f"  [SKIP] Last entry was {mins} min ago — 1-hour gap not reached yet.")
                print(f"         This is why no new row was inserted.")
            else:
                print(f"  [PASS] Last entry was {mins} min ago — gap is fine, insert should work.")
        else:
            print("  [PASS] No previous entry — gap check will pass.")

    # Step 6: Force a direct INSERT and check result
    print("\n[INFO] Step 6: Attempting a direct test INSERT for 'Sasidhar'...")
    cursor.execute("SELECT person_id FROM Persons WHERE name = ?", ("Sasidhar",))
    row = cursor.fetchone()
    if not row:
        print("  [SKIP] Cannot insert — 'Sasidhar' not found in Persons table.")
    else:
        person_id = row[0]
        try:
            cursor.execute(
                "INSERT INTO Attendance (person_id, arrival_time, status) VALUES (?, ?, 'INSIDE')",
                (person_id, datetime.now())
            )
            conn.commit()
            print(f"  [PASS] Direct INSERT succeeded! Rows affected: {cursor.rowcount}")
            print(f"         Check SQL Server now — you should see a new row.")

            # Clean it up
            cursor.execute(
                "DELETE FROM Attendance WHERE person_id = ? AND status = 'INSIDE' "
                "AND arrival_time > ?",
                (person_id, datetime.now() - timedelta(seconds=10))
            )
            conn.commit()
            print(f"  [INFO] Test row cleaned up.")
        except Exception as e:
            print(f"  [FAIL] INSERT failed — {e}")

    conn.close()
    print("\n========== DIAGNOSTIC COMPLETE ==========\n")

if __name__ == "__main__":
    run()