from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from data_loader import DataLoader
from memory import MemoryManager
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI

class RAGSystem:
    def __init__(self, model="openai"):
        self.model = model
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
        Analyze using these data sources:
        {context}

        Conversation History:
        {chat_history}

        Follow these rules:
        1. Prioritize wearable data for health queries
        2. Use location data for time-based questions
        3. Maintain professional tone

        Question: {question}
        Answer:"""
        )

    def _get_llm(self):
       self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
