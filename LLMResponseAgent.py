import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class LLMResponseAgent:

    def __init__(self, retriever):

        prompt_template = """
        Use the context below to answer the question. If you don't know the answer from the context, just say you don't know.
        Context: {context}
        Question: {question}
        Helpful Answer:
        """
        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        self.llm = GoogleGenerativeAI(temperature=0, model="models/gemini-2.0-flash")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def run(self, query: str) -> dict:
        print(f"[LLMResponseAgent] Answering question: '{query}'")
        return self.qa_chain.invoke(query)