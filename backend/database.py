import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote_plus

import asyncpg
from dotenv import load_dotenv

_backend_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_backend_dir)
load_dotenv(os.path.join(_repo_root, ".env"))
load_dotenv(os.path.join(_backend_dir, ".env"))


def get_database_dsn() -> str:
    """
    Resolve Postgres DSN.
    - DATABASE_URL (non-empty): use as-is (Neon, RDS, etc.).
    - Else POSTGRES_PASSWORD: local postgres://postgres:PASSWORD@127.0.0.1:5432/faceliveness
    - Else: passwordless local URL (works only if pg_hba allows trust/peer for local).
    """
    explicit = (os.getenv("DATABASE_URL") or "").strip()
    if explicit:
        return explicit
    pw = (os.getenv("POSTGRES_PASSWORD") or "").strip()
    if pw:
        return f"postgresql://postgres:{quote_plus(pw)}@127.0.0.1:5432/faceliveness"
    return "postgresql://postgres@127.0.0.1:5432/faceliveness"


_pool: Optional[asyncpg.Pool] = None


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "jsonb",
        encoder=lambda value: json.dumps(value, default=_json_default),
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=lambda value: json.dumps(value, default=_json_default),
        decoder=json.loads,
        schema="pg_catalog",
    )


async def init_db_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        database_url = get_database_dsn()
        try:
            _pool = await asyncpg.create_pool(
                dsn=database_url,
                min_size=int(os.getenv("DB_POOL_MIN_SIZE", "1")),
                max_size=int(os.getenv("DB_POOL_MAX_SIZE", "10")),
                command_timeout=float(os.getenv("DB_COMMAND_TIMEOUT", "30")),
                init=_init_connection,
            )
        except Exception as e:
            err_name = type(e).__name__
            if err_name == "InvalidPasswordError" or "password authentication failed" in str(e).lower():
                print(
                    "\n❌ PostgreSQL: password authentication failed.\n"
                    "   Set POSTGRES_PASSWORD in .env to your local 'postgres' user password (from pgAdmin),\n"
                    "   or set DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/faceliveness\n"
                )
            raise
    return _pool


async def close_db_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        return await init_db_pool()
    return _pool


@asynccontextmanager
async def transaction():
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn


async def execute(query: str, *args: Any) -> str:
    pool = await get_pool()
    return await pool.execute(query, *args)


async def fetch(query: str, *args: Any) -> List[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetch(query, *args)


async def fetchrow(query: str, *args: Any) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetchrow(query, *args)


async def ensure_app_users_table() -> None:
    await execute(
        """
        CREATE TABLE IF NOT EXISTS app_users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            face_label TEXT NOT NULL,
            image_url TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    await execute(
        "CREATE INDEX IF NOT EXISTS idx_app_users_email ON app_users (email)"
    )


async def ensure_face_register_users_table() -> None:
    """
    Separate table for frontend registered users with 512-d embedding.
    Kept in sync from `faces` so existing insert/match logic remains unchanged.
    """
    await execute(
        """
        CREATE TABLE IF NOT EXISTS face_register_users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            face_label TEXT NOT NULL,
            image_url TEXT,
            embedding JSONB NOT NULL,
            source TEXT NOT NULL DEFAULT 'frontend_reg',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT face_register_users_embedding_is_array
                CHECK (jsonb_typeof(embedding) = 'array'),
            CONSTRAINT face_register_users_embedding_len_512
                CHECK (jsonb_array_length(embedding) = 512)
        )
        """
    )
    await execute(
        "CREATE INDEX IF NOT EXISTS idx_face_register_users_email_lower ON face_register_users (LOWER(email))"
    )
    await execute(
        "CREATE INDEX IF NOT EXISTS idx_face_register_users_face_label_lower ON face_register_users (LOWER(face_label))"
    )
    await execute(
        """
        INSERT INTO face_register_users (id, email, face_label, image_url, embedding, source, created_at, updated_at)
        SELECT
            f.id,
            LOWER(TRIM(f.label)) AS email,
            f.label AS face_label,
            f.image_url,
            f.embedding,
            COALESCE(f.source, 'frontend_reg') AS source,
            NOW(),
            NOW()
        FROM faces f
        WHERE f.embedding IS NOT NULL
          AND jsonb_typeof(f.embedding) = 'array'
          AND jsonb_array_length(f.embedding) = 512
          AND LOWER(TRIM(COALESCE(f.source, ''))) = 'frontend_reg'
          AND POSITION('@' IN COALESCE(f.label, '')) > 0
        ON CONFLICT (email) DO UPDATE
        SET
            id = EXCLUDED.id,
            face_label = EXCLUDED.face_label,
            image_url = EXCLUDED.image_url,
            embedding = EXCLUDED.embedding,
            source = EXCLUDED.source,
            updated_at = NOW()
        """
    )
    await execute(
        """
        CREATE OR REPLACE FUNCTION sync_face_register_users_from_faces()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                IF LOWER(TRIM(COALESCE(OLD.source, ''))) = 'frontend_reg'
                   AND POSITION('@' IN COALESCE(OLD.label, '')) > 0 THEN
                    DELETE FROM face_register_users
                    WHERE email = LOWER(TRIM(OLD.label));
                END IF;
                RETURN OLD;
            END IF;

            IF LOWER(TRIM(COALESCE(NEW.source, ''))) = 'frontend_reg'
               AND POSITION('@' IN COALESCE(NEW.label, '')) > 0
               AND NEW.embedding IS NOT NULL
               AND jsonb_typeof(NEW.embedding) = 'array'
               AND jsonb_array_length(NEW.embedding) = 512 THEN

                INSERT INTO face_register_users (
                    id, email, face_label, image_url, embedding, source, created_at, updated_at
                )
                VALUES (
                    NEW.id,
                    LOWER(TRIM(NEW.label)),
                    NEW.label,
                    NEW.image_url,
                    NEW.embedding,
                    COALESCE(NEW.source, 'frontend_reg'),
                    NOW(),
                    NOW()
                )
                ON CONFLICT (email) DO UPDATE
                SET
                    id = EXCLUDED.id,
                    face_label = EXCLUDED.face_label,
                    image_url = EXCLUDED.image_url,
                    embedding = EXCLUDED.embedding,
                    source = EXCLUDED.source,
                    updated_at = NOW();
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    await execute(
        "DROP TRIGGER IF EXISTS trg_sync_face_register_users_from_faces ON faces"
    )
    await execute(
        """
        CREATE TRIGGER trg_sync_face_register_users_from_faces
        AFTER INSERT OR UPDATE OR DELETE ON faces
        FOR EACH ROW
        EXECUTE FUNCTION sync_face_register_users_from_faces()
        """
    )


async def email_exists(email: str) -> bool:
    clean = email.strip().lower()
    if not clean:
        return False

    # New dedicated table for frontend-registered users (primary source).
    row = await fetchrow(
        "SELECT 1 FROM face_register_users WHERE LOWER(email) = LOWER($1) LIMIT 1",
        clean,
    )
    if row is not None:
        return True

    # Backward compatibility with existing tables.
    row = await fetchrow(
        "SELECT 1 FROM app_users WHERE LOWER(email) = LOWER($1) LIMIT 1",
        clean,
    )
    if row is not None:
        return True

    row = await fetchrow(
        """
        SELECT 1
        FROM faces
        WHERE LOWER(label) = LOWER($1) AND embedding IS NOT NULL
        LIMIT 1
        """,
        clean,
    )
    return row is not None


async def resolve_registered_face_label(email: str) -> Optional[str]:
    """
    Return registered face_label for login/liveness/match.
    Checks face_register_users first, then app_users, then faces.label fallback.
    """
    e = email.strip().lower()
    if not e or "@" not in e:
        return None

    fru_row = await fetchrow(
        """
        SELECT face_label FROM face_register_users
        WHERE LOWER(email) = LOWER($1)
        LIMIT 1
        """,
        e,
    )
    if fru_row and fru_row.get("face_label"):
        return str(fru_row["face_label"]).strip()

    user_row = await fetchrow(
        """
        SELECT face_label FROM app_users
        WHERE LOWER(email) = LOWER($1)
        LIMIT 1
        """,
        e,
    )
    if user_row and user_row.get("face_label"):
        return str(user_row["face_label"]).strip()

    face_row = await fetchrow(
        """
        SELECT label FROM faces
        WHERE LOWER(label) = LOWER($1) AND embedding IS NOT NULL
        ORDER BY id
        LIMIT 1
        """,
        e,
    )
    if face_row and face_row.get("label"):
        return str(face_row["label"]).strip()

    return None


async def is_registered_email(email: str) -> bool:
    return await resolve_registered_face_label(email) is not None


async def insert_app_user(
    *,
    email: str,
    full_name: str,
    face_label: str,
    image_url: Optional[str] = None,
) -> None:
    await execute(
        """
        INSERT INTO app_users (id, email, full_name, face_label, image_url)
        VALUES ($1, $2, $3, $4, $5)
        """,
        str(uuid.uuid4()),
        email.strip().lower(),
        full_name.strip(),
        face_label,
        image_url,
    )


async def ensure_indexes() -> None:
    await ensure_app_users_table()
    await ensure_face_register_users_table()
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_faces_label ON faces (label)",
        "CREATE INDEX IF NOT EXISTS idx_faces_source ON faces (source)",
        "CREATE INDEX IF NOT EXISTS idx_liveness_sessions_session_id ON liveness_sessions (session_id)",
        "CREATE INDEX IF NOT EXISTS idx_liveness_sessions_device_sequence ON liveness_sessions (device_id, sequence_key)",
        "CREATE INDEX IF NOT EXISTS idx_liveness_sessions_status_expires ON liveness_sessions (status, expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_liveness_sessions_raw_data_gin ON liveness_sessions USING GIN (raw_data)",
        "CREATE INDEX IF NOT EXISTS idx_auth_logs_logged_at ON auth_logs (logged_at)",
        "CREATE INDEX IF NOT EXISTS idx_auth_logs_top_match ON auth_logs (top_match)",
        "CREATE INDEX IF NOT EXISTS idx_auth_logs_raw_data_gin ON auth_logs USING GIN (raw_data)",
    ]
    for statement in statements:
        await execute(statement)


async def list_face_labels() -> List[str]:
    rows = await fetch(
        """
        SELECT DISTINCT label
        FROM faces
        WHERE label IS NOT NULL AND label <> 'txt'
        ORDER BY label
        """
    )
    return [row["label"] for row in rows]


async def get_faces_by_labels(labels: Iterable[str]) -> List[Dict[str, Any]]:
    label_list = list(labels)
    if not label_list:
        return []
    rows = await fetch(
        """
        SELECT label, image_url
        FROM faces
        WHERE label = ANY($1::text[])
        """,
        label_list,
    )
    return [dict(row) for row in rows]


async def get_face_embedding_by_label(label: str) -> Optional[List[float]]:
    row = await fetchrow(
        """
        SELECT embedding
        FROM faces
        WHERE label = $1 AND embedding IS NOT NULL
        ORDER BY id
        LIMIT 1
        """,
        label,
    )
    return row["embedding"] if row else None


async def list_faces_for_matching() -> List[Dict[str, Any]]:
    rows = await fetch(
        """
        SELECT label, source, image_url, embedding
        FROM faces
        WHERE embedding IS NOT NULL
        """
    )
    return [dict(row) for row in rows]


async def list_faces_for_matching_by_label(label: str) -> List[Dict[str, Any]]:
    """Embeddings for one identity only (logged-in 1:1 verification)."""
    clean = (label or "").strip()
    if not clean:
        return []
    rows = await fetch(
        """
        SELECT label, source, image_url, embedding
        FROM faces
        WHERE embedding IS NOT NULL
          AND LOWER(TRIM(label)) = LOWER(TRIM($1))
        """,
        clean,
    )
    return [dict(row) for row in rows]


async def insert_face(
    *,
    label: str,
    source: str,
    image_url: str,
    embedding: List[float],
) -> None:
    await execute(
        """
        INSERT INTO faces (id, label, source, image_url, embedding)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        """,
        str(uuid.uuid4()),
        label,
        source,
        image_url,
        embedding,
    )


async def _execute_once(query: str, *args: Any) -> str:
    conn = await asyncpg.connect(dsn=get_database_dsn())
    await _init_connection(conn)
    try:
        return await conn.execute(query, *args)
    finally:
        await conn.close()


def execute_sync(query: str, *args: Any) -> str:
    return asyncio.run(_execute_once(query, *args))


def insert_face_sync(
    *,
    label: str,
    source: str,
    image_url: str,
    embedding: List[float],
) -> str:
    return execute_sync(
        """
        INSERT INTO faces (id, label, source, image_url, embedding)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        """,
        str(uuid.uuid4()),
        label,
        source,
        image_url,
        embedding,
    )


async def create_liveness_session(
    *,
    session_id: str,
    device_id: str,
    gestures: List[str],
    status: str,
    expires_at: datetime,
    raw_data: Optional[Dict[str, Any]] = None,
) -> None:
    sequence_key = "|".join(gestures)
    await execute(
        """
        INSERT INTO liveness_sessions (
            id, session_id, device_id, gestures, sequence_key, status, expires_at, created_at, raw_data
        )
        SELECT $1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9::jsonb
        WHERE NOT EXISTS (
            SELECT 1 FROM liveness_sessions WHERE device_id = $3 AND sequence_key = $5
        )
        """,
        str(uuid.uuid4()),
        session_id,
        device_id,
        gestures,
        sequence_key,
        status,
        expires_at,
        datetime.utcnow(),
        raw_data or {},
    )


async def complete_liveness_session_if_valid(
    *,
    session_id: str,
    now: datetime,
    raw_updates: Optional[Dict[str, Any]] = None,
) -> bool:
    result = await execute(
        """
        UPDATE liveness_sessions
        SET status = 'completed',
            raw_data = COALESCE(raw_data, '{}'::jsonb) || $3::jsonb
        WHERE session_id = $1
          AND status = 'issued'
          AND expires_at > $2
        """,
        session_id,
        now,
        raw_updates or {},
    )
    return result == "UPDATE 1"


async def get_valid_completed_liveness_session(
    *,
    session_id: str,
    device_id: str,
    now: datetime,
) -> Optional[Dict[str, Any]]:
    row = await fetchrow(
        """
        SELECT session_id, device_id, status, expires_at, raw_data
        FROM liveness_sessions
        WHERE session_id = $1
          AND device_id = $2
          AND status = 'completed'
          AND expires_at > $3
        LIMIT 1
        """,
        session_id,
        device_id,
        now,
    )
    return dict(row) if row else None


async def get_liveness_session(
    *,
    session_id: str,
    device_id: str,
) -> Optional[Dict[str, Any]]:
    row = await fetchrow(
        """
        SELECT session_id, device_id, status, expires_at, raw_data
        FROM liveness_sessions
        WHERE session_id = $1
          AND device_id = $2
        LIMIT 1
        """,
        session_id,
        device_id,
    )
    return dict(row) if row else None


async def update_liveness_session_status(
    *,
    session_id: str,
    status: str,
    raw_updates: Optional[Dict[str, Any]] = None,
) -> None:
    await execute(
        """
        UPDATE liveness_sessions
        SET status = $2,
            raw_data = COALESCE(raw_data, '{}'::jsonb) || $3::jsonb
        WHERE session_id = $1
        """,
        session_id,
        status,
        raw_updates or {},
    )


def parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_optional_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.utcnow()


async def insert_auth_log(
    *,
    timestamp: Any,
    geo_lat: Optional[str],
    geo_long: Optional[str],
    top_match: str,
    match_count: int,
    raw_data: Optional[Dict[str, Any]] = None,
) -> None:
    await execute(
        """
        INSERT INTO auth_logs (id, timestamp, geo_lat, geo_long, top_match, match_count, logged_at, raw_data)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
        """,
        str(uuid.uuid4()),
        parse_optional_datetime(timestamp),
        geo_lat if geo_lat else None,
        geo_long if geo_long else None,
        top_match,
        match_count,
        datetime.utcnow(),
        raw_data or {},
    )


async def count_faces() -> Dict[str, int]:
    row = await fetchrow(
        """
        SELECT
            COUNT(*)::int AS total,
            COUNT(*) FILTER (WHERE embedding IS NOT NULL)::int AS with_embedding
        FROM faces
        """
    )
    valid_rows = await fetch("SELECT embedding FROM faces WHERE embedding IS NOT NULL")
    with_512 = sum(1 for row in valid_rows if isinstance(row["embedding"], list) and len(row["embedding"]) == 512)
    return {
        "total": row["total"] if row else 0,
        "with_embedding": row["with_embedding"] if row else 0,
        "with_512": with_512,
    }
