"""User accounts stored in SQLite.

SQLite is a full SQL database in a single file: no server to run, ideal for
a demo, and the ``UserStore`` interface means we could move to Postgres
later without changing callers.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.auth.passwords import DUMMY_HASH, hash_password, verify_password

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    full_name     TEXT NOT NULL,
    role          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    disabled      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True)
class User:
    username: str
    full_name: str
    role: str
    disabled: bool = False


class UserAlreadyExistsError(ValueError):
    pass


class UserStore:
    def __init__(self, db_path: Path | str) -> None:
        """``db_path`` may be ':memory:' for tests."""
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # One connection; check_same_thread=False because FastAPI runs sync
        # endpoints in a thread pool. SQLite serialises writes itself.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._conn:  # commits on success, rolls back on exception
            yield self._conn

    def close(self) -> None:
        self._conn.close()

    # ---------- queries ----------

    def get(self, username: str) -> User | None:
        row = self._conn.execute(
            # Parameterised query (?): user input is never pasted into SQL,
            # which makes SQL injection impossible here.
            "SELECT username, full_name, role, disabled FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        return User(row["username"], row["full_name"], row["role"], bool(row["disabled"]))

    def list_users(self) -> list[User]:
        rows = self._conn.execute(
            "SELECT username, full_name, role, disabled FROM users ORDER BY username"
        ).fetchall()
        return [User(r["username"], r["full_name"], r["role"], bool(r["disabled"])) for r in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def authenticate(self, username: str, password: str) -> User | None:
        """Return the user if the credentials are valid and the account is active."""
        row = self._conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        stored_hash = row["password_hash"] if row else DUMMY_HASH
        password_ok = verify_password(password, stored_hash)  # always runs: equal timing
        if row is None or not password_ok:
            return None
        user = self.get(username)
        return user if user and not user.disabled else None

    # ---------- commands ----------

    def create(self, username: str, full_name: str, role: str, password: str) -> User:
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO users (username, full_name, role, password_hash) "
                    "VALUES (?, ?, ?, ?)",
                    (username, full_name, role, hash_password(password)),
                )
        except sqlite3.IntegrityError as exc:
            raise UserAlreadyExistsError(username) from exc
        return User(username, full_name, role)

    def upsert(self, username: str, full_name: str, role: str, password: str) -> User:
        """Create or overwrite a user (used for idempotent seeding)."""
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO users (username, full_name, role, password_hash) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(username) DO UPDATE SET full_name=excluded.full_name, "
                "role=excluded.role, password_hash=excluded.password_hash, disabled=0",
                (username, full_name, role, hash_password(password)),
            )
        return User(username, full_name, role)

    def set_disabled(self, username: str, disabled: bool) -> None:
        with self._transaction() as conn:
            conn.execute(
                "UPDATE users SET disabled = ? WHERE username = ?", (int(disabled), username)
            )

    def set_role(self, username: str, role: str) -> None:
        with self._transaction() as conn:
            conn.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
