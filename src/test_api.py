import os
from dotenv import load_dotenv
import cfbd

load_dotenv()
key = os.getenv("CFBD_API_KEY")
if not key:
    raise SystemExit("No API key found. Check your .env file.")

config = cfbd.Configuration(access_token=key)
with cfbd.ApiClient(config) as client:
    games = cfbd.GamesApi(client).get_games(year=2025, week=1)

print(f"Connected. Pulled {len(games)} games from 2025 Week 1.")
for g in games[:5]:
    print(f"{g.away_team} @ {g.home_team}: {g.away_points}-{g.home_points}")