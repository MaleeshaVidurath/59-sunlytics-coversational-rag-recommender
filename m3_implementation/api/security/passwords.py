"""
Password hashing and policy.

Argon2id — OWASP's first choice for password storage. Memory-hard, so GPU and
ASIC attacks gain far less than they do against bcrypt or PBKDF2.
"""
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.profiles import RFC_9106_LOW_MEMORY

from .config import PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH

# RFC 9106's low-memory profile (64 MiB, t=3, p=4) is the recommended choice
# for a server that also has an LLM pipeline competing for RAM. The parameters
# are recorded inside each hash, so raising them later does not invalidate
# existing passwords — they rehash on next successful login.
_hasher = PasswordHasher.from_parameters(RFC_9106_LOW_MEMORY)

# A precomputed hash of a value nobody will guess. Verifying against this on an
# unknown username makes a "no such user" response cost the same as a wrong
# password, so response timing does not reveal which accounts exist.
_DUMMY_HASH = _hasher.hash("dummy-password-for-constant-time-comparison")

# Passwords that dominate every breach corpus. A production system would check
# against a full k-anonymity breach API; this list covers the trivial cases
# without a network dependency.
_COMMON_PASSWORDS = {
    "password", "password1", "password123", "123456", "12345678", "123456789",
    "1234567890", "qwerty", "qwerty123", "abc123", "letmein", "welcome",
    "admin", "administrator", "iloveyou", "monkey", "dragon", "sunshine",
    "princess", "football", "baseball", "trustno1", "passw0rd", "p@ssw0rd",
    "changeme", "secret", "111111", "000000", "654321", "superman",
    "qwertyuiop", "asdfghjkl", "zxcvbnm", "1q2w3e4r", "1qaz2wsx",
    "sunlytics", "sunlytics123", "hm2024", "fashion123",
}


class PasswordPolicyError(ValueError):
    """Raised when a proposed password fails policy. Message is user-safe."""


def validate_password(password: str, *, username: str | None = None) -> None:
    """
    Enforces the password policy, raising PasswordPolicyError on the first
    problem. Length first — it is the control that actually matters.
    """
    if not password or not password.strip():
        raise PasswordPolicyError("Password cannot be empty.")

    if len(password) < PASSWORD_MIN_LENGTH:
        raise PasswordPolicyError(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
        )

    # Argon2 has no practical input limit, but an unbounded password is a cheap
    # way to make the server burn CPU on every login attempt.
    if len(password) > PASSWORD_MAX_LENGTH:
        raise PasswordPolicyError(
            f"Password must be at most {PASSWORD_MAX_LENGTH} characters."
        )

    lowered = password.lower()

    # Exact match, and also the common case of a weak base with digits or
    # punctuation bolted on: "password123456" and "Passw0rd!!" are no stronger
    # than "password". Stripping the non-alphabetic edges catches those without
    # rejecting a genuine passphrase that merely contains a common word.
    stripped = lowered.strip("0123456789!@#$%^&*()_+-=[]{};:,.<>?/|\'\"`~ ")
    if lowered in _COMMON_PASSWORDS or stripped in _COMMON_PASSWORDS:
        raise PasswordPolicyError(
            "That password is too common. Choose something less predictable."
        )

    if username and username.lower() in lowered:
        raise PasswordPolicyError("Password must not contain your username.")


def hash_password(password: str) -> str:
    """Returns an Argon2id hash string; the salt and parameters live inside it."""
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str | None) -> bool:
    """
    Checks a password against a stored hash.

    Passing None (an account that does not exist) still performs a real verify
    against a dummy hash, so the caller cannot leak account existence through
    how long the request took.
    """
    target = stored_hash or _DUMMY_HASH
    try:
        _hasher.verify(target, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    # Never report success for an account that has no credential on file.
    return stored_hash is not None


def needs_rehash(stored_hash: str) -> bool:
    """True when a hash was made with weaker parameters than we now use."""
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True
