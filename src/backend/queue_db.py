import json
import sqlite3
import time
from typing import Optional, Any


def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            result TEXT,
            payload_json_path TEXT
        )
        """
    )
    
    # Migration: Add payload_json_path column if it doesn't exist
    try:
        cur.execute("SELECT payload_json_path FROM queue LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        print("Migrating queue database: adding payload_json_path column")
        cur.execute("ALTER TABLE queue ADD COLUMN payload_json_path TEXT")
    
    conn.commit()
    conn.close()


def enqueue_job(db_path: str, payload: Any, payload_json_path: str = None) -> int:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now = time.time()
    cur.execute(
        "INSERT INTO queue (payload, status, created_at, payload_json_path) VALUES (?, ?, ?, ?)",
        (json.dumps(payload), "queued", now, payload_json_path),
    )
    job_id = cur.lastrowid
    conn.commit()
    conn.close()
    return job_id


def get_job(db_path: str, job_id: int) -> Optional[dict]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM queue WHERE id = ?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def list_jobs(db_path: str, status: Optional[str] = None) -> list:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if status:
        cur.execute("SELECT * FROM queue WHERE status = ? ORDER BY created_at DESC", (status,))
    else:
        cur.execute("SELECT * FROM queue ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def pop_next_job(db_path: str) -> Optional[dict]:
    """Atomically claim the next queued job and mark it as running. Returns the job row or None."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("SELECT * FROM queue WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1")
        row = cur.fetchone()
        if not row:
            conn.commit()
            return None
        job = dict(row)
        now = time.time()
        cur.execute(
            "UPDATE queue SET status = ?, started_at = ? WHERE id = ?",
            ("running", now, job["id"]),
        )
        conn.commit()
        return job
    finally:
        conn.close()


def complete_job(db_path: str, job_id: int, result: Any):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now = time.time()
    cur.execute(
        "UPDATE queue SET status = ?, finished_at = ?, result = ? WHERE id = ?",
        ("done", now, json.dumps(result), job_id),
    )
    conn.commit()
    conn.close()


def fail_job(db_path: str, job_id: int, error: str):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now = time.time()
    cur.execute(
        "UPDATE queue SET status = ?, finished_at = ?, result = ? WHERE id = ?",
        ("failed", now, json.dumps({"error": error}), job_id),
    )
    conn.commit()
    conn.close()


def cancel_job(db_path: str, job_id: int) -> bool:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT status FROM queue WHERE id = ?", (job_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False
    status = row[0]
    # Can only cancel queued jobs, not running/done/failed/cancelled
    if status != "queued":
        conn.close()
        return False
    cur.execute("UPDATE queue SET status = ? WHERE id = ?", ("cancelled", job_id))
    conn.commit()
    conn.close()
    return True


def reset_orphaned_jobs(db_path: str) -> int:
    """Reset any 'running' jobs to 'failed' on startup (orphaned by container restart/crash).
    Returns the count of jobs reset."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now = time.time()
    # Find all running jobs
    cur.execute("SELECT id FROM queue WHERE status = 'running'")
    rows = cur.fetchall()
    count = len(rows)
    if count > 0:
        # Mark them as failed with appropriate message
        cur.execute(
            "UPDATE queue SET status = ?, finished_at = ?, result = ? WHERE status = 'running'",
            ("failed", now, json.dumps({"error": "Job interrupted by container restart"})),
        )
        conn.commit()
    conn.close()
    return count
