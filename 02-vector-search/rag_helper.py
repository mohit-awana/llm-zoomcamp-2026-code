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
from ingest import vec_to_str


@dataclass
class RAGResultRagBase:
    answer: str
    usage: object
    response: object


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


class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)

        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )

    
class RAGPgVector(RAGBase):

    def __init__(self, embedder, conn, course, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn
        self.course = course

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]
    
    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(
                f""""
                Course: {doc["course"]}
                Section: {doc["section"]}
                Question: {doc["question"]}
                Answer: {doc["answer"]}
                """
                )
            return "\n\n".join(lines)