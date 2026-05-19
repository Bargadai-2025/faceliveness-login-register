import asyncio
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from database import close_db_pool, fetch, init_db_pool


async def main():
    await init_db_pool()
    try:
        rows = await fetch(
            """
            SELECT id, label, source, image_url, embedding
            FROM faces
            WHERE label = 'Angel Mary'
            ORDER BY id
            """
        )
    finally:
        await close_db_pool()

    vecs = []
    for r in rows:
        v = np.array(r["embedding"], dtype="float32")
        v = v / (np.linalg.norm(v) + 1e-6)
        vecs.append(v)
        print(r["source"], r["image_url"][:90], "...")

    if len(vecs) >= 2:
        sim = float(np.dot(vecs[0], vecs[1]))
        print(f"\nCosine similarity between Angel Mary DB photos: {sim:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
