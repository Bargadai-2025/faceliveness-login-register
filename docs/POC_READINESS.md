# PoC / Pilot Readiness (Banking Demo)

## Deploy checklist

- [ ] `FAST_MATCH=0`, `LIVENESS_FAST_SETUP=0`
- [ ] `RISK_REJECT_THRESHOLD=76`, `RISK_CONFIRM_FACTORS_MIN=3`
- [ ] HTTPS via nginx (`backend/nginx-faceliveliness.example.conf`)
- [ ] `VITE_API_URL` points to HTTPS API prefix
- [ ] Health: `GET /health/live`, `GET /health/ready`
- [ ] Optional: `REDIS_URL` for session survival across restarts / single replica recovery

## What was hardened

| Area | Implementation |
|------|----------------|
| Sessions | `session_store.py` — memory + Redis pickle sync |
| Concurrency | `inference_runtime.py` — bounded ML thread pool; `embedding_cache.py` — in-memory vector search |
| Logging | `poc_logging.py` — JSON security pass/reject events |
| API errors | `api_errors.py` — no raw exceptions to clients |
| Rate limits | `rate_limit.py` — per-IP on frame/match/start |
| Retries | Backend `retry_allowed` + frontend 2 selfie retries |
| UX | No mid-challenge security card on soft display hints |
| Match | Accepts liveness step `verified` or `complete` |

## Demo test matrix

1. Real user — window / tube light → should complete + match
2. Low-end Android — full 3 gestures + selfie
3. Weak network — match retry toast (up to 2×)
4. Phone screen replay — reject after multi-factor (may retry once if borderline)
5. Server restart with Redis — session can resume; without Redis — restart liveness

## Monitoring

Search logs for JSON events: `security_match_pass`, `security_match_reject`, `redis_session_store_ready`.
