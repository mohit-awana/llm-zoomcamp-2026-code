INSTRUCTIONS = """
Your task is to answer questions from the course lessons
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

AGENTIC_INSTRUCTIONS = """
You're a course teaching assistant.
Answer the student's question using the search tool.
Make multiple searches with different keywords before answering.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:  {context}
""".strip()


from dataclasses import dataclass
from typing import Callable
from ingest import vec_to_str

from pydantic_ai import Agent


@dataclass
class RAGResultRagBase:
    answer: str
    usage: object
    response: object

@dataclass
class RAGResult:
    answer: str
    usage: object
    response: object
    search_calls: object


class RAGBase():
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="gpt-5.4-mini",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {
            "content": 3.0,
            "filename": 0.5,
        }

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            filename = doc.get("filename", "").strip()
            content = doc.get("content", "").strip()

            if filename:
                lines.append(filename)
            if content:
                lines.append(content)
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )

        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return RAGResultRagBase(
            answer=response.output_text,
            usage=response.usage,
            response=response,
        )


def build_search_tool(index) -> Callable[[str], list[dict]]:
    def search(query: str) -> list[dict]:
        """Search the lesson chunk index for relevant passages.

        Args:
            query: Search keywords to look up in the course lessons.
        """
        boost_dict = {
            "content": 3.0,
            "filename": 0.5,
        }

        return index.search(
            query,
            num_results=5,
            boost_dict=boost_dict,
        )

    return search


class AgenticRAG:
    def __init__(
        self,
        index,
        instructions=AGENTIC_INSTRUCTIONS,
        model="openai-responses:gpt-5.4-mini",
    ):
        self.index = index #Store the search index.
        self.instructions = instructions #Store system prompt.
        self.model = model #Store model name.

    async def rag(self, query):

        # 1. Initialize a mutable counter
        call_tracker = {"count": 0}

        base_search_fn = build_search_tool(self.index)

        # 2. Create a wrapper function that increments the counter
        def search_fn(query: str) -> list[dict]:
            """Search the lesson chunk index for relevant passages.

            Args:
                query: Search keywords to look up in the course lessons.
            """
            call_tracker["count"] += 1  # Increment on every call
            return base_search_fn(query)
        
        agent = Agent(
            self.model,
            instructions=self.instructions,
            tools=[search_fn],
        )
        result = await agent.run(query)
        usage = getattr(result, "usage", None)
        if callable(usage):
            usage = usage()
        # 3. Access the total count via call_tracker["count"]
        print(f"The search tool was called {call_tracker['count']} times.")

        return RAGResult(
            answer=result.output,
            usage=usage,
            response=result,
            search_calls = call_tracker["count"]
        )
