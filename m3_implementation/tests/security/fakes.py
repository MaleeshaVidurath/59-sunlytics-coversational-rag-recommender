"""
In-memory MongoDB and Redis stand-ins for the security tests.

Shared so the two flow tests do not each carry their own copy — a fake that
drifts between test files is worse than no fake, because the two suites then
silently exercise different behaviour.

A caution learned the hard way: these fakes are not MongoDB. Real Mongo returns
BSON dates as *naive* datetimes, while a value stored here stays timezone-aware.
That difference hid a bug where `AccountDocument.is_locked` raised TypeError on
every request from a locked-out account. Test model behaviour against naive
datetimes directly — do not trust the fake to reproduce it.
"""
import asyncio
import re

import httpx


class FakeCollection:
    def __init__(self):
        self.docs = []
        self.unique = set()

    def _match(self, doc, query):
        for key, value in query.items():
            if isinstance(value, dict):
                if "$regex" in value:
                    if not re.match(value["$regex"], str(doc.get(key, ""))):
                        return False
                elif "$ne" in value:
                    if doc.get(key) == value["$ne"]:
                        return False
                elif "$in" in value:
                    if doc.get(key) not in value["$in"]:
                        return False
                elif "$exists" in value:
                    if (key in doc) != value["$exists"]:
                        return False
            elif doc.get(key) != value:
                return False
        return True

    async def create_index(self, key, **kwargs):
        if kwargs.get("unique"):
            self.unique.add(key)

    async def count_documents(self, query):
        return sum(1 for d in self.docs if self._match(d, query))

    async def find_one(self, query, projection=None):
        return next((dict(d) for d in self.docs if self._match(d, query)), None)

    async def insert_one(self, doc):
        for key in self.unique:
            if any(d.get(key) == doc.get(key) for d in self.docs):
                raise Exception(f"duplicate key: {key}")
        self.docs.append(dict(doc))

    async def insert_many(self, docs):
        for doc in docs:
            await self.insert_one(doc)

    async def update_one(self, query, update):
        for doc in self.docs:
            if self._match(doc, query):
                doc.update(update.get("$set", {}))
                for key, delta in update.get("$inc", {}).items():
                    doc[key] = doc.get(key, 0) + delta
                return _Result(modified_count=1)
        return _Result(modified_count=0)

    async def update_many(self, query, update):
        count = 0
        for doc in self.docs:
            if self._match(doc, query):
                doc.update(update.get("$set", {}))
                count += 1
        return _Result(modified_count=count)

    async def delete_one(self, query):
        for i, doc in enumerate(self.docs):
            if self._match(doc, query):
                self.docs.pop(i)
                return _Result(deleted_count=1)
        return _Result(deleted_count=0)

    async def delete_many(self, query):
        before = len(self.docs)
        self.docs = [d for d in self.docs if not self._match(d, query)]
        return _Result(deleted_count=before - len(self.docs))

    async def find_one_and_update(self, query, update, **kwargs):
        await self.update_one(query, update)
        return await self.find_one(query)

    def find(self, query, projection=None):
        return _FakeCursor([dict(d) for d in self.docs if self._match(d, query)])


class _Result:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def __aiter__(self):
        async def gen():
            for doc in self.docs:
                yield doc
        return gen()

    async def to_list(self, length=None):
        return self.docs


class FakeDB:
    def __init__(self):
        self._collections = {}

    def __getattr__(self, name):
        return self._collections.setdefault(name, FakeCollection())

    def __getitem__(self, name):
        return self._collections.setdefault(name, FakeCollection())


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]

    async def expire(self, key, seconds):
        pass

    async def ttl(self, key):
        return 60

    async def delete(self, key):
        self.data.pop(key, None)
        return 1


class Client:
    """
    Synchronous facade over httpx.ASGITransport.

    Starlette's own TestClient does not work with httpx 0.28, which this project
    pins for the Groq calls. Uses https:// because the auth cookies are marked
    Secure and httpx correctly refuses to send those over http — browsers make an
    exception for localhost, httpx does not.
    """

    def __init__(self, app, loop):
        self._loop = loop
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://testserver")

    @property
    def cookies(self):
        return self._client.cookies

    def _csrf_headers(self):
        token = self._client.cookies.get("sunlytics_csrf")
        return {"X-CSRF-Token": token} if token else {}

    def get(self, url, **kwargs):
        return self._loop.run_until_complete(self._client.get(url, **kwargs))

    def post(self, url, csrf=True, **kwargs):
        """`csrf=False` omits the header, so tests can prove it is required."""
        if csrf:
            kwargs.setdefault("headers", {}).update(self._csrf_headers())
        return self._loop.run_until_complete(self._client.post(url, **kwargs))

    def delete(self, url, csrf=True, **kwargs):
        if csrf:
            kwargs.setdefault("headers", {}).update(self._csrf_headers())
        return self._loop.run_until_complete(self._client.delete(url, **kwargs))


def install_fakes():
    """
    Points the app's database accessors at the in-memory fakes.

    Must run before the routers are imported, since they capture get_db at
    import time.
    """
    db, redis = FakeDB(), FakeRedis()

    import memory.db.mongo as mongo_mod
    import memory.db.redis_client as redis_mod
    mongo_mod.get_db = lambda: db
    mongo_mod.get_collection_name = lambda name: name
    redis_mod.get_redis = lambda: redis

    import api.security.models as sec_models
    sec_models.get_db = lambda: db
    import api.security.rate_limit as rl_mod
    rl_mod.get_redis = lambda: redis

    return db, redis


def new_loop():
    return asyncio.new_event_loop()
