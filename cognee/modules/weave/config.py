import os


def get_internal_token() -> str:
    """Return the private Amberly-to-Weave bearer token.

    The token is deliberately loaded per request so rotations do not require a
    module reload. Empty tokens are never accepted.
    """

    return os.getenv("WEAVE_INTERNAL_TOKEN", "")
