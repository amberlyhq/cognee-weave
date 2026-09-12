"""Legacy repository lifecycle predicate, retained for compatibility tests."""

def repository_accepts_review(lifecycle, generation):
    return bool(
        lifecycle
        and lifecycle.active
        and not getattr(lifecycle, "deletion_pending", False)
        and lifecycle.lifecycle_generation == generation
    )
