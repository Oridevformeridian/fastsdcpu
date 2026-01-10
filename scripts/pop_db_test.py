#!/usr/bin/env python3
"""Pop one job via backend.queue_db.pop_next_job and immediately revert it to 'queued'.
This script is safe to run: it reverts the change so the queue is not altered.
"""
import json
import sqlite3
import sys
from pathlib import Path

# ensure project on sys.path
# add both project root and src dir so imports like `backend.*` resolve
sys.path.insert(0, '/app/fastsdcpu/src')
sys.path.insert(0, '/app/fastsdcpu')

DB = '/app/fastsdcpu/results/queue.db'

try:
    from backend.queue_db import pop_next_job
except Exception as e:
    print('IMPORT_ERROR', str(e))
    raise

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
try:
    job = pop_next_job(DB)
    if not job:
        print('NO_JOB_POPPED')
        sys.exit(0)
    print('POPPED')
    print(json.dumps(dict(job)))
    # revert
    cur = conn.cursor()
    cur.execute("UPDATE queue SET status='queued', started_at=NULL WHERE id=?", (job['id'],))
    conn.commit()
    print('REVERTED', job['id'])
except Exception as e:
    print('ERROR', str(e))
    raise
finally:
    conn.close()
