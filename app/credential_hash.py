import hashlib
import hmac
import secrets


_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 600_000
_MIN_VERIFY_ITERATIONS = 100_000
_MAX_VERIFY_ITERATIONS = 1_000_000
_SALT_BYTES = 16


def hash_credential(value: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        value.encode("utf-8"),
        salt,
        _ITERATIONS,
    )
    return f"{_ALGORITHM}${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_credential(value: str, encoded: str) -> tuple[bool, bool]:
    """Return whether the credential matches and whether a legacy hash matched."""
    if encoded.startswith(f"{_ALGORITHM}$"):
        try:
            algorithm, iteration_text, salt_hex, digest_hex = encoded.split("$")
            iterations = int(iteration_text)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(digest_hex)
        except (ValueError, TypeError):
            return False, False

        if (
            algorithm != _ALGORITHM
            or not _MIN_VERIFY_ITERATIONS <= iterations <= _MAX_VERIFY_ITERATIONS
            or len(salt) != _SALT_BYTES
            or len(expected) != hashlib.sha256().digest_size
        ):
            return False, False

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            value.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual, expected), False

    if len(encoded) != hashlib.sha256().digest_size * 2:
        return False, False
    try:
        bytes.fromhex(encoded)
    except ValueError:
        return False, False

    legacy_digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    matches = hmac.compare_digest(legacy_digest, encoded)
    return matches, matches
