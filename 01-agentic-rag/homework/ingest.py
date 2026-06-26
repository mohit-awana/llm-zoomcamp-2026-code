from minsearch import Index
from gitsource import GithubRepositoryDataReader, chunk_documents
import requests


def load_faq_data():
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = 'https://datatalks.club/faq'

    for course in courses_raw:
        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents

def load_lesson_data():
    reader = GithubRepositoryDataReader(repo_owner="DataTalksClub"
                                        ,repo_name="llm-zoomcamp"
                                        ,commit_id="8c1834d"
                                        ,allowed_extensions={"md"}
                                        ,filename_filter=lambda path: "/lessons/" in path,
                                        )
    files = reader.read()
    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)
    
    return documents

def chunk(documents):
    
    chunks = chunk_documents(documents, size=2000, step=1000)
    
    return chunks

def build_index(chunks):
    
    index = Index(text_fields = ["content"],
                keyword_fields = ["filename"]
                )
    index.fit(chunks)
    return index



#We want a chunker that is:

#1.Semantic → doesn't cut in the middle of sentences.
#2.Token-aware → respects LLM token limits.
#token: a chunk of characters—averaging four characters or about three-quarters of a word—that serves as the fundamental atomic unit the model reads, processes, and generates
#3.Overlapping → repeats some context between chunks.


import re
from typing import List, Dict
import tiktoken

import re 
from typing import List, Dict
import tiktoken 


class SemanticTokenChunker:

    def __init__(
        self,
        max_tokens: int = 500, ##too large wil degarde the reterival process 
        overlap_tokens: int = 50, ##usualy between 10-20% of max_token to start with
        encoding_name: str = "cl100k_base", ##OpenAI tokenizer
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def split_sentences(self, text: str) -> List[str]:
        return [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', text) #(?<=...) is a positive lookbehind. It checks that a pattern exists immediately before the current position without consuming it.
                                                      # [.!?] key charac filter 
                                                      #\s+ - match one or more whitespace characters (spaces, tabs, newlines).
            if s.strip() #filters out empty sentences so the returned list contains only clean, non-empty sentences.
        ]
    
    def chunk(self, text: Dict) -> List[Dict]:

        sentences = self.split_sentences(text['content'])
        filename = text['filename']

        chunks = []

        current_sentences = [] #stores the sentences being accumulated for the current chunk.
        current_tokens = 0 #tracks the total tokens in that chunk

        for sentence in sentences:

            sentence_tokens = self.token_count(sentence)

            if (current_tokens + sentence_tokens > self.max_tokens and current_sentences):
                #When current_tokens + sentence_tokens > max_tokens, the current chunk is complete, so the sentences are joined and saved to chunks before starting a new chunk.
                chunk_text = " ".join(current_sentences)

                chunks.append(
                    {
                        "content": chunk_text,
                        "tokens": current_tokens,
                        "filename": filename
                    }
                )

                # token-aware overlap
                overlap = []

                overlap_tokens = 0

                #Create overlap for the next chunk
                for s in reversed(current_sentences):

                    s_tokens = self.token_count(s)

                    if (overlap_tokens + s_tokens > self.overlap_tokens):
                        break

                    overlap.insert(0, s) #adds the sentence s at the beginning of the overlap list, preserving the original sentence order
                    overlap_tokens += s_tokens
                
                current_sentences = overlap
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(
                {
                    "content": " ".join(current_sentences),
                    "tokens": current_tokens,
                    "filename": filename
                }
            )

        return chunks
