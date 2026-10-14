import duckdb

con = duckdb.connect("data/cfb.duckdb", read_only=True)

print("\nGAMES PER SEASON")
print(con.execute("""
    SELECT pull_year AS season,
           pull_season_type AS type,
           COUNT(*) AS total_games,
           SUM(CASE WHEN homeClassification = 'fbs' OR awayClassification = 'fbs' THEN 1 ELSE 0 END) AS fbs_involved,
           SUM(CASE WHEN homePoints IS NOT NULL THEN 1 ELSE 0 END) AS completed
    FROM games
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df().to_string(index=False))

print("\nCLASSIFICATIONS IN GAMES TABLE")
print(con.execute("""
    SELECT homeClassification, COUNT(*) AS n
    FROM games GROUP BY 1 ORDER BY 2 DESC
""").df().to_string(index=False))

print("\n2026 STATUS")
print(con.execute("""
    SELECT MAX(week) AS last_completed_week
    FROM games
    WHERE pull_year = 2026 AND homePoints IS NOT NULL
""").df().to_string(index=False))

con.close()