import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from database import close_db_pool, fetch, init_db_pool


async def main():
    await init_db_pool()
    try:
        rows = await fetch(
            """
            SELECT label, source, COUNT(*)::int AS n
            FROM faces
            WHERE LOWER(label) LIKE '%angel%' OR LOWER(label) LIKE '%mary%'
            GROUP BY label, source
            ORDER BY label
            """
        )
        for r in rows:
            print(dict(r))
    finally:
        await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())
