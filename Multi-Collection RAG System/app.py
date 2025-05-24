import chainlit as chain
from rag import RAGSystem
from memory import MemoryManager

@chain.on_chat_start
async def init_chat():
    session_id = chain.user_session.get("id")
    rag_system = RAGSystem()
    memory = MemoryManager(session_id).get_memory()
    
    chain.user_session.set("chain", rag_system.create_chain(memory))
    
    await chain.Message(
        content="Welcome to Chatbot!\n"
        "I can access:\n"
        "- Wearable Data\n- Chat History\n- User Profile\n"
        "- Location Data\n- Custom Goals"
    ).send()

@chain.on_message
async def main(message: chain.Message):
    chain = chain.user_session.get("chain")
    response = await chain.acall(message.content)
    

    source = list({doc.metadata["collection"] for doc in response["source_documents"]})
    response = f"{response['answer']}\n\nSources: {', '.join(source)}"
    
    await chain.Message(content=response).send()