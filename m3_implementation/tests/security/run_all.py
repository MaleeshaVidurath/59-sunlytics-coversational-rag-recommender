"""
Runs the security suite.

    python m3_implementation/tests/security/run_all.py

Each test file runs in its own process. That is deliberate: they patch module
globals (get_db, get_redis) at import time, so sharing an interpreter would let
one suite's fakes leak into another's.

No databases are required — everything runs against in-memory fakes.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SUITES = [
    ("primitives",    "test_primitives.py",    "hashing, tokens, policy, cookie flags"),
    ("auth flow",     "test_auth_flow.py",     "register, login, refresh rotation, CSRF, lockout"),
    ("authorization", "test_authorization.py", "session ownership, the original vulnerability"),
]


def main() -> int:
    failed = []

    for name, filename, summary in SUITES:
        print("=" * 70)
        print(f"{name}  —  {summary}")
        print("=" * 70)
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, filename)],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            failed.append(name)
        print()

    print("=" * 70)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(SUITES)} security suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
