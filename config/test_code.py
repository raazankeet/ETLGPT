import pymysql, json, os

DB="iics"
conn = pymysql.connect(
    host="192.168.0.108",
    port=3306,
    user="grafanauser",
    password="ankitrajdba",
    database=DB,
    cursorclass=pymysql.cursors.Cursor,
)
text_like = {"char","varchar","text","mediumtext","longtext"}
embed, skip = {}, []
with conn.cursor() as cur:
    cur.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM information_schema.columns
        WHERE table_schema=%s
    """, (DB,))
    for tbl, col, dtype in cur.fetchall():
        if dtype.lower() not in text_like:
            skip.append(f"{tbl}.{col}")  # numeric/dates default to skip for value embeddings
            continue
        cur.execute(f"SELECT COUNT(DISTINCT `{col}`), COUNT(*) FROM `{tbl}`")
        distinct, total = cur.fetchone()
        if total == 0 or distinct == 0:
            continue
        ratio = distinct / total
        # heuristics: embed if moderate variety (not almost unique)
        if 0 < ratio < 0.4 and distinct <= 500:
            embed.setdefault(tbl, []).append(col)
        else:
            skip.append(f"{tbl}.{col}")
print("embed_values candidates:", json.dumps(embed, indent=2))
print("skip_values candidates:", json.dumps(skip, indent=2))
