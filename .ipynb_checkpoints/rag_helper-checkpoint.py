INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

#

#We use a class because `index` and `openai_client` are currently global
#variables. Move the functions to a separate file and those globals
#aren't there anymore. We could import them back, but that ties the file
#to one specific index and one specific client. That makes the code hard
#to reuse and adjust.

#So we put the dependencies inside a class instead. The index and the
#LLM client become constructor arguments. Now we can pass any index or
#client we want when we create the object. And because it's a class, we
#can subclass it later to override one piece without touching the rest.
#For example, we can swap OpenAI for a local model.

#

class RAGBase():
    def __init__(
        self,
        index,
        llm_client,
        instructions = INSTRUCTIONS,
        prompt_template = PROMPT_TEMPLATE,
        course = "llm-zoomcamp",
        model = "gpt-5.4-nano"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
#The `index` parameter is anything with a `search` method, whether
#minsearch, sqlitesearch, or something else. The other four parameters
#all have defaults. You only pass `course`, `instructions`,
#`prompt_template`, or `model` when you want to override the default
#behavior. We swap the index later without touching any of the RAG code.

#The `search` method delegates to the index:

    
    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

#The `build_context` and `build_prompt` methods format the search results:
    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

#The `llm` method sends the prompt to the LLM:

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

#And the `rag` method wires it all together:
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer