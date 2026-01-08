import sqlite3
import os
from typing import Optional, Dict


def _get_conn(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: str) -> None:
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                note TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def set_review(db_path: str, name: str, status: str, note: Optional[str]) -> None:
    init_db(db_path)
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO reviews(name, status, note) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET status=excluded.status, note=excluded.note",
            (name, status, note),
        )
        conn.commit()
    finally:
        conn.close()


def get_review(db_path: str, name: str) -> Optional[Dict[str, str]]:
    if not os.path.exists(db_path):
        return None
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT status, note FROM reviews WHERE name = ?", (name,))
        row = cur.fetchone()
        if not row:
            return None
        return {"status": row[0], "note": row[1]}
    finally:
        conn.close()


def delete_review(db_path: str, name: str) -> bool:
    if not os.path.exists(db_path):
        return False
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM reviews WHERE name = ?", (name,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_reviews(db_path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(db_path):
        return {}
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, status, note FROM reviews")
        rows = cur.fetchall()
        return {r[0]: {"status": r[1], "note": r[2]} for r in rows}
    finally:
        conn.close()
