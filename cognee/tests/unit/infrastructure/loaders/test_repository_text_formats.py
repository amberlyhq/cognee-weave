"""Repository text files use native loaders without executing their contents."""

import pytest
from cognee.infrastructure.loaders import get_loader_engine
from cognee.tasks.code_graph.code_repo import partition_repo_files

FORMATS = {
    "guide.mdx": "# Setup\n<Component />\n",
    "schema.sql": "CREATE TABLE orders (id uuid PRIMARY KEY);\n",
    "settings.toml": "enabled = true\n",
    "service.conf": "port=3000\n",
    "script.sh": "#!/bin/sh\necho hello\n",
    "styles.css": ".button { color: blue; }\n",
    "worker.mjs": "export const active = true;\n",
    "worker.cjs": "module.exports = { active: true };\n",
    "service.dockerfile": "FROM scratch\n",
    "change.patch": "--- a/app.ts\n+++ b/app.ts\n",
    "diagram.svg": '<svg xmlns="http://www.w3.org/2000/svg"><text>Orders</text></svg>',
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name,content", FORMATS.items())
async def test_repository_text_format_is_selected_and_read_verbatim(tmp_path, name, content):
    (tmp_path / "package.json").write_text("{}")
    path = tmp_path / name
    path.write_text(content)
    _, documents, skipped = partition_repo_files(tmp_path)
    assert path in documents
    assert path not in skipped
    loader = get_loader_engine().get_loader(str(path), preferred_loaders=None)
    assert loader.loader_name == "text_loader"
    assert await loader.load(str(path), persist=False) == content


def test_repository_binary_and_secret_files_remain_excluded(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / ".env").write_text("SECRET=not-for-indexing")
    (tmp_path / "asset.bin").write_bytes(b"\x00\xff\x00")
    _, documents, skipped = partition_repo_files(tmp_path)
    assert documents == []
    assert {p.name for p in skipped} == {".env", "asset.bin"}
