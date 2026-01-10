#!/usr/bin/env python3
"""Pop one job via backend.queue_db.pop_next_job and print it; do NOT revert.
This will set the job status to 'running' in the DB.
"""
import json
import sqlite3
import sys

# ensure imports resolve when run inside container
sys.path.insert(0, '/app/fastsdcpu/src')
sys.path.insert(0, '/app/fastsdcpu')

DB = '/app/fastsdcpu/results/queue.db'

try:
    from backend.queue_db import pop_next_job
except Exception as e:
    print('IMPORT_ERROR', str(e))
    raise

job = pop_next_job(DB)
if not job:
    print('NO_JOB_POPPED')
    sys.exit(0)
print('POPPED')
print(json.dumps(dict(job)))
