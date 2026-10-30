# 9.3 Prompt Templates (Halaman 103-104)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Simple template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])

# Format to messages (for inspection)
msgs = prompt.format_messages(
    role="Python tutor",
    language="English",
    question="What is a generator?",
)
print("Formatted simple prompt:", msgs)

# Template with conversation history placeholder
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

formatted = chat_prompt.format_messages(
    context="User is learning about RAG",
    history=[
        HumanMessage(content="What is retrieval?"),
    ],
    input="How is it different from fine-tuning?",
)
print(f"Total messages in formatted prompt: {len(formatted)}")
