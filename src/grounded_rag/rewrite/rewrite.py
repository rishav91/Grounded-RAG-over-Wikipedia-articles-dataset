"""`rewrite_query`: one LLM call that decontextualizes/expands the query and
decomposes it into independent sub-queries when genuinely applicable (FR11;
ADR-011).

A plain function, not a LangGraph node — mirrors `generate.py`/`rerank.py`/
`check_sufficiency`'s split between the plain function and its graph wrapper
in `graph/nodes.py`.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from grounded_rag.config import QUERY_REWRITE_MAX_SUB_QUERIES
from grounded_rag.rewrite.prompts import SYSTEM_PROMPT, build_rewrite_prompt
from grounded_rag.rewrite.records import RewriteResult


class RewriteJudgment(BaseModel):
    rewritten_query: str
    sub_queries: list[str]


def rewrite_query(llm: BaseChatModel, query: str) -> RewriteResult:
    try:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_rewrite_prompt(query))]
        judgment = llm.with_structured_output(RewriteJudgment).invoke(messages)
    except Exception:
        # Fail open: a transient rewrite-LLM outage must never block
        # retrieval — mirrors ADR-010's tier-2 fallback (check_sufficiency).
        return RewriteResult(rewritten_query=query, sub_queries=[])

    return RewriteResult(
        rewritten_query=judgment.rewritten_query or query,
        sub_queries=judgment.sub_queries[:QUERY_REWRITE_MAX_SUB_QUERIES],
        rewritten_by_llm=True,
    )
