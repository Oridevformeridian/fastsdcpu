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
            payload_json_path TEXT,
            progress TEXT,
            retry_count INTEGER DEFAULT 0
        )
        """
    )
    
    # Create settings table for queue control (pause, etc.)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS queue_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
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
    
    # Migration: Add progress column if it doesn't exist
    try:
        cur.execute("SELECT progress FROM queue LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating queue database: adding progress column")
        cur.execute("ALTER TABLE queue ADD COLUMN progress TEXT")
    
    # Migration: Add retry_count column if it doesn't exist
    try:
        cur.execute("SELECT retry_count FROM queue LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating queue database: adding retry_count column")
        cur.execute("ALTER TABLE queue ADD COLUMN retry_count INTEGER DEFAULT 0")
    
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


def update_job_progress(db_path: str, job_id: int, progress_data: dict):
    """Update job progress for checkpoint tracking"""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE queue SET progress = ? WHERE id = ?",
        (json.dumps(progress_data), job_id),
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
    """Reset any 'running' jobs back to 'queued' for retry on startup (orphaned by container restart/crash).
    Returns the count of jobs reset."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Find all running jobs
    cur.execute("SELECT id, retry_count FROM queue WHERE status = 'running'")
    rows = cur.fetchall()
    count = len(rows)
    if count > 0:
        # Reset to queued for retry, increment retry count
        for job_id, retry_count in rows:
            new_retry_count = (retry_count or 0) + 1
            if new_retry_count > 3:
                # Too many retries, mark as failed
                cur.execute(
                    "UPDATE queue SET status = ?, finished_at = ?, result = ?, retry_count = ? WHERE id = ?",
                    ("failed", time.time(), json.dumps({"error": f"Job failed after {new_retry_count} attempts (interrupted by restarts)"}), new_retry_count, job_id),
                )
            else:
                # Reset to queued for retry
                cur.execute(
                    "UPDATE queue SET status = ?, started_at = NULL, progress = ?, retry_count = ? WHERE id = ?",
                    ("queued", json.dumps({"note": f"Retry #{new_retry_count} after restart"}), new_retry_count, job_id),
                )
        conn.commit()
    conn.close()
    return count


def is_queue_paused(db_path: str) -> bool:
    """Check if the queue is paused."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT value FROM queue_settings WHERE key = 'paused'")
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return row[0] == "true"


def set_queue_paused(db_path: str, paused: bool):
    """Set the queue pause state."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    value = "true" if paused else "false"
    cur.execute(
        "INSERT OR REPLACE INTO queue_settings (key, value) VALUES (?, ?)",
        ("paused", value),
    )
    conn.commit()
    conn.close()
