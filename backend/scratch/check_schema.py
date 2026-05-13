import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def check_schema():
    database_url = os.getenv("DATABASE_URL") or "postgresql://postgres:localhost@localhost:5432/faceliveness"
    conn = await asyncpg.connect(database_url)
    try:
        rows = await conn.fetch("""
            SELECT column_name, data_type, column_default, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'liveness_sessions'
        """)
        for row in rows:
            print(f"{row['column_name']}: {row['data_type']} (default: {row['column_default']}, nullable: {row['is_nullable']})")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_schema())
