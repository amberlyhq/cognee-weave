> Historical design. The current implementation is [agent memory storage](agent-memory-storage.md): governance owns the agents and Weave only applies their selected native-memory operations.

# Code graph first, review memory later

Repository imports use native Cognee code extraction with vector indexing and
self-improvement disabled. Enola builds code relationships without sending the
whole repository through an LLM memory-building pass.

Code and qualified review facts share one native dataset per customer. Repository
identity scopes source records, file links and retrieval inside that dataset.
See [shared customer graph](shared-customer-graph.md) for migration and merge behavior.

Completed reviews supply their results and bounded saved agent activity. The
configured model makes one qualification pass to select useful facts, their
certainty, code paths and source passages. It distinguishes historical review
claims from current implementation. There is no second model auditor, static
semantic-quality checker, or code-driven certainty rewriting.

The application attaches the selected source passages and validates source IDs,
quote provenance and repository-relative paths. Invalid references go back to the
same qualification step for bounded correction; facts with valid references are
retained. Semantic support and usefulness remain the model's responsibility.
A successful write does not prove that the model's interpretation is correct.

Native remember receives selected facts and source references. Full source quotes
remain in receipts and graph annotations. Each write processes only its own source
document, leaving old documents and code archives untouched. Durable receipts
allow repeated deliveries to reuse completed work. The configured native adapter,
model, provider routing and embeddings are unchanged.

On default-branch changes, native code indexing refreshes the graph first. A merge
qualification pass uses the indexed exact-commit source to retain, supersede or
leave prior facts unverified, and to add useful new facts. Historical facts remain
stored. Current replacements link to repository files and their source documents.

Tests cover single-pass qualification, reference correction, native Postgres
storage and recall, duplicate delivery, A-B-A indexing, deletion and customer
isolation. Fixture model responses test integration behavior; staging runs test
live model behavior separately.
