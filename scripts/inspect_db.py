import sqlite3

conn = sqlite3.connect("data/metadata.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)

for table_name in ("documents", "chunks"):
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    print(f"\n{table_name} columns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

conn.close()