from __future__ import annotations

import json
from typing import Any

from elasticsearch import Elasticsearch

from elasticsearch_backend import ElasticsearchFAQIndex


TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": "Search the FAQ database for entries matching the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
]

search_tool = TOOLS[0]

FAQ_INDEX: ElasticsearchFAQIndex | None = None


def build_index(
    client: Elasticsearch,
    *,
    index_name: str = "faq",
    course: str = "llm-zoomcamp",
) -> ElasticsearchFAQIndex:
    """Create a reusable Elasticsearch FAQ index wrapper."""

    return ElasticsearchFAQIndex(client=client, index_name=index_name, course=course)


def configure_index(index: ElasticsearchFAQIndex) -> None:
    """Register the index used by ``search(query)`` and tool dispatch."""

    global FAQ_INDEX
    FAQ_INDEX = index


def search(query: str) -> list[dict[str, Any]]:
    """Search the FAQ index with the boosts and filter used in the course."""

    if FAQ_INDEX is None:
        raise RuntimeError("FAQ_INDEX is not configured. Call configure_index() first.")

    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return FAQ_INDEX.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
    )


def make_call(call: Any) -> dict[str, Any]:
    """Dispatch an OpenAI function call to the local Elasticsearch search."""

    arguments = json.loads(call.arguments)

    if call.name != "search":
        raise ValueError(f"Unsupported tool call: {call.name}")

    result = search(arguments["query"])
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": json.dumps(result),
    }
