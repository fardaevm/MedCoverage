from __future__ import annotations
import os
import json
from hashlib import sha1
from typing import Optional
import redis


class RedisJSONCache:
    def __init__(self):
        url = (os.getenv("REDIS_URL") or "").strip()
        if not url:
            url = "redis://redis:6379/0"  # docker-compose service name
        self.url = url

        self.ttl = int(os.getenv("QUESTIONS_CACHE_TTL_SECONDS", "604800"))
        self.client: Optional[redis.Redis] = None
        self.last_error: Optional[str] = None

        try:
            c = redis.Redis.from_url(
                self.url,
                decode_responses=True,
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
                health_check_interval=15,
            )
            c.ping()
            self.client = c
        except Exception as e:
            self.client = None
            self.last_error = f"{type(e).__name__}: {e}"

    def enabled(self) -> bool:
        return self.client is not None

    @staticmethod
    def make_key(prefix: str, payload: str) -> str:
        h = sha1(payload.encode("utf-8")).hexdigest()
        return f"{prefix}:{h}"

    def get(self, key: str) -> Optional[dict]:
        if not self.client:
            return None
        try:
            raw = self.client.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, key: str, value: dict):
        if not self.client:
            return
        try:
            self.client.setex(key, self.ttl, json.dumps(value, ensure_ascii=False))
        except Exception:
            pass