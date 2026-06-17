from __future__ import annotations

from typing import Any, Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ElasticsearchFAQIndex:
    """Elasticsearch-backed FAQ index compatible with ``RAGBase``.

    The class keeps the same ``search()`` interface as the in-memory/minsearch
    backends already used in this repo, so it can be dropped into the existing
    RAG pipeline without changing ``rag_helper.py``.
    """

    def __init__(
        self,
        client: Elasticsearch,
        index_name: str = "faq",
        course: str = "llm-zoomcamp",
        text_fields: tuple[str, ...] = ("question", "answer", "section"),
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.course = course
        self.text_fields = text_fields

    def create_index(self, recreate: bool = False) -> None:
        if recreate and self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)

        if self.client.indices.exists(index=self.index_name):
            return

        mappings: dict[str, Any] = {
            "properties": {
                "course": {"type": "keyword"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "answer": {"type": "text"},
            }
        }

        self.client.indices.create(index=self.index_name, mappings=mappings)

    def add_documents(self, documents: Iterable[dict[str, Any]]) -> None:
        actions = (
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_source": document,
            }
            for document in documents
        )
        bulk(self.client, actions)
        self.client.indices.refresh(index=self.index_name)

    def count(self) -> int:
        if not self.client.indices.exists(index=self.index_name):
            return 0
        response = self.client.count(index=self.index_name)
        return int(response["count"])

    def _build_filters(self, filter_dict: dict[str, Any] | None) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []

        merged_filters = {"course": self.course}
        if filter_dict:
            merged_filters.update(filter_dict)

        for field, value in merged_filters.items():
            if isinstance(value, (list, tuple, set)):
                filters.append({"terms": {field: list(value)}})
            else:
                filters.append({"term": {field: value}})

        return filters

    def search(
        self,
        query: str,
        num_results: int = 5,
        boost_dict: dict[str, float] | None = None,
        filter_dict: dict[str, Any] | None = None,
        multi_match_type: str = "best_fields",
    ) -> list[dict[str, Any]]:
        boost_dict = boost_dict or {"question": 3.0, "answer": 2.0, "section": 0.5}
        fields = [f"{field}^{boost}" if boost != 1 else field for field, boost in boost_dict.items()]

        query_body: dict[str, Any] = {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": fields,
                        "type": multi_match_type,
                    }
                },
                "filter": self._build_filters(filter_dict),
            }
        }

        response = self.client.search(
            index=self.index_name,
            size=num_results,
            query=query_body,
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]
