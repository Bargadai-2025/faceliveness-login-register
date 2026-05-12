import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import close_db_pool, count_faces, init_db_pool

load_dotenv()

async def main():
    await init_db_pool()
    try:
        counts = await count_faces()
    finally:
        await close_db_pool()

    print(f"Total rows: {counts['total']}")
    print(f"Rows with any embedding: {counts['with_embedding']}")
    print(f"Rows with valid 512-dim embedding: {counts['with_512']}")


if __name__ == "__main__":
    asyncio.run(main())
