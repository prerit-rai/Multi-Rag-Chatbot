from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory

class MemoryManager:
    def __init__(self, session_id):
        self.session_id = session_id
        self.message_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string="sqlite:///memory.db"
        )
        
        self.buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=self.message_history,
            return_messages=True
        )

    def get_memory(self):
        return self.buffer_memory