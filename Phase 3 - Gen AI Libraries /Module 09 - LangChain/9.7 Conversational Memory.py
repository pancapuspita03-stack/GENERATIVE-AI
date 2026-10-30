# 9.7 Conversational Memory (Halaman 110-112)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3.5-super-vl:free",
    api_key=os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key")
)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | parser

# Session store - keyed by session_id
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

if __name__ == "__main__":
    config = {"configurable": {"session_id": "user-123"}}
    # Turn 1
    r1 = chain_with_history.invoke({"input": "What is RAG?"}, config=config)
    print("Turn 1:", r1[:100])
    # Turn 2 - model remembers turn 1
    r2 = chain_with_history.invoke({"input": "What are the main components I just asked about?"}, config=config)
    print("Turn 2:", r2[:100])
    # Inspect stored history
    history = get_session_history("user-123")
    print(f"Messages in history: {len(history.messages)}")
