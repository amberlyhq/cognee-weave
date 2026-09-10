"""Keep the native repository identity stable across disposable GitHub archives."""


def prepare_repository_directory(repository, request):
    # Native CodeLoader identifies a project by its directory basename.
    stable = repository.parent / (
        f"{request.repository_owner}-{request.repository_name}-{request.github_repository_id}"
    )
    return repository if repository == stable else repository.rename(stable)
