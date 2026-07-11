"""Prompt for `rewrite_query` (FR11; ADR-011).

The independence instruction is the load-bearing part of this prompt: it's
what keeps `UC-4`-style sequential multi-hop (the second hop's subject is
only known after reading the first hop's chunk) from being silently routed
away from FR8's reactive tool-call path and into a fabricated, entity-less
sub-query that the retrieval index won't match.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You rewrite a search query to improve retrieval, before \
any document has been retrieved. You do not answer the question.

Produce:
- `rewritten_query`: the question restated as a clear, standalone search \
query — resolve vague phrasing and add terms that would help a search \
engine match the right article, without changing what's being asked. If \
the question is already clear and specific, this may be the same as the \
original.
- `sub_queries`: a list of additional, standalone search queries, ONLY if \
the question bundles multiple parts that can each be answered by an \
INDEPENDENT search — each sub-query must be fully self-contained and \
searchable on its own, right now, without needing an answer from any other \
part first.

Leave `sub_queries` empty (the common case) whenever:
- The question is single-part.
- The question is multi-hop but SEQUENTIAL: a later part refers to an \
entity or fact ("that person", "its successor", "the son mentioned above") \
that can only be identified by first retrieving and reading an earlier \
part's answer. Do NOT invent a sub-query that uses a placeholder or \
unresolved reference for that entity — an unresolved reference is not a \
valid standalone search query. Sequential multi-hop questions are handled \
by a different mechanism after you decide not to decompose them here.

Only populate `sub_queries` when each part genuinely stands alone — e.g. \
"What is [Article-A] known for, and what is [Article-B] known for?" \
decomposes cleanly because each half names its own subject already."""


def build_rewrite_prompt(query: str) -> str:
    return f"Question: {query}"
