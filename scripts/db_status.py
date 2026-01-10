#!/usr/bin/env python3
import sqlite3
DB='/app/fastsdcpu/results/queue.db'
conn=sqlite3.connect(DB)
cur=conn.cursor()
cur.execute("SELECT status, COUNT(*) FROM queue GROUP BY status")
rows=cur.fetchall()
for r in rows:
    print(r[0], r[1])
conn.close()
