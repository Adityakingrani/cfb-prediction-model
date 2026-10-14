import duckdb

con = duckdb.connect("data/cfb.duckdb", read_only=True)

print("\n2026 WEEKS")
print(con.execute("""
    SELECT week,
           COUNT(*) AS games,
           SUM(CASE WHEN homePoints IS NOT NULL THEN 1 ELSE 0 END) AS completed
    FROM games
    WHERE pull_year = 2026
      AND homeClassification = 'fbs' AND awayClassification = 'fbs'
    GROUP BY week ORDER BY week
""").df().to_string(index=False))

for table in ["games", "adv_game_stats", "sp_ratings", "returning_production"]:
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    print(f"\n{table.upper()} ({len(cols)} columns)")
    print(", ".join(cols))

con.close()