from types import SimpleNamespace
from uuid import uuid4

import pytest


def fixture():
    org = uuid4()
    binding = SimpleNamespace(organization_id=org, dataset_id=uuid4())
    receipt = SimpleNamespace(
        organization_id=org,
        github_repository_id=42,
        source_key="review:qualified:" + str(uuid4()),
        data_id=uuid4(),
        dataset_id=binding.dataset_id,
        artifact_revision=0,
        qualification={
            "facts": [
                {
                    "statement": "Reviewer proposes requiring stable replay IDs.",
                    "code_path": "src/Pay.ts",
                    "certainty": "reported",
                    "evidence": [
                        {
                            "evidence_id": "final",
                            "quote": "Reviewer proposes requiring stable replay IDs.",
                        }
                    ],
                }
            ]
        },
    )

    def file(org=org, repo=42, path="src/Pay.ts"):
        return str(uuid4()), {
            "type": "CodeFileReference",
            "organization_id": str(org),
            "github_repository_id": repo,
            "file_path": path,
            "indexed_sha": "a" * 40,
        }

    return binding, receipt, file


def test_links_preserve_reported_proposal_and_exact_tenant_repository_path():
    from cognee.modules.weave.review_code_links import build_review_links

    binding, receipt, file = fixture()
    correct = file()
    nodes, edges = build_review_links(
        binding,
        receipt,
        "b" * 40,
        [correct, file(org=uuid4()), file(repo=43), file(path="src/pay.ts")],
    )
    assert len(nodes) == 1
    assert nodes[0].certainty == "reported"
    assert nodes[0].statement.startswith("Reviewer proposes")
    assert nodes[0].reviewed_head_sha == "b" * 40
    assert [(str(a), str(b), r) for a, b, r, _ in edges] == [
        (str(nodes[0].id), correct[0], "review_context_for")
    ]
    assert edges[0][3]["historical"] is True
    assert edges[0][3]["certainty"] == "reported"


def test_missing_file_keeps_fact_unlinked_instead_of_guessing():
    from cognee.modules.weave.review_code_links import build_review_links

    binding, receipt, file = fixture()
    nodes, edges = build_review_links(binding, receipt, "b" * 40, [file(repo=43)])
    assert len(nodes) == 1 and edges == []


@pytest.mark.parametrize("field", ["organization_id", "dataset_id"])
def test_foreign_receipt_cannot_create_links(field):
    from cognee.modules.weave.review_code_links import build_review_links

    binding, receipt, file = fixture()
    setattr(receipt, field, uuid4())
    with pytest.raises(ValueError, match="Foreign"):
        build_review_links(binding, receipt, "b" * 40, [file()])


def test_fact_lifecycle_is_mirrored_on_node_and_link():
    from cognee.modules.weave.review_code_links import build_review_links
    from cognee.modules.weave.knowledge_lifecycle import fact_digest

    binding, receipt, file = fixture()
    digest = fact_digest(receipt.qualification["facts"][0])
    receipt.qualification["fact_states"] = {
        digest: {
            "status": "needs_recheck",
            "checked_sha": "a" * 40,
            "invalidated_sha": "b" * 40,
        }
    }
    nodes, edges = build_review_links(binding, receipt, "a" * 40, [file()])
    assert nodes[0].knowledge_status == "needs_recheck"
    assert nodes[0].checked_sha == "a" * 40
    assert edges[0][3]["knowledge_status"] == "needs_recheck"
