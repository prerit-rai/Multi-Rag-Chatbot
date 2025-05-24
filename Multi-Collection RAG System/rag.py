from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from data_loader import DataLoader
from memory import MemoryManager
from langchain.chat_models import ChatOpenAI

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = self._initialize_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

    def _initialize_vector_store(self):
        loader = DataLoader()
        docs = loader.load_documents()
        return FAISS.from_documents(docs, self.embeddings)

    def create_chain(self, memory):
        return ConversationalRetrievalChain.from_llm(
            llm=self._get_llm(),
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": self._create_prompt()}
        )

    def _create_prompt(self):
        from langchain.prompts import PromptTemplate
        return PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
            Context from collections:
            {context}
            
            Conversation History:
            {chat_history}
            
            Question: {question}
            Answer:"""
        )

    def _get_llm(self):
       return ChatOpenAI(
        model_name="gpt-4-0125-preview",
        temperature=0.3,
        max_tokens=400
    )