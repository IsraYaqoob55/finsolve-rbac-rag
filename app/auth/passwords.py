"""Password hashing with bcrypt.

We never store passwords, only a bcrypt hash. bcrypt is deliberately slow
and salted, so a stolen database can't be reversed with lookup tables and
brute force is expensive.
"""

import bcrypt

_BCRYPT_MAX_BYTES = 72  # bcrypt ignores input beyond 72 bytes; newer versions raise
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    encoded = password.encode("utf-8")
    if len(encoded) > _BCRYPT_MAX_BYTES:
        raise ValueError(f"password must be at most {_BCRYPT_MAX_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    encoded = password.encode("utf-8")
    if len(encoded) > _BCRYPT_MAX_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("ascii"))
    except ValueError:  # malformed hash in the database
        return False


# A real hash of a random string. When a username doesn't exist we still run
# one bcrypt check against this, so "unknown user" and "wrong password" take
# the same time and attackers can't discover valid usernames by timing.
DUMMY_HASH = bcrypt.hashpw(b"timing-equaliser-not-a-password", bcrypt.gensalt()).decode("ascii")
