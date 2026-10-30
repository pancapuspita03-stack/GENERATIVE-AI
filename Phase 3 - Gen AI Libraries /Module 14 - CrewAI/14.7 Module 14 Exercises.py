# 14.7 Module 14 Exercises (Halaman 163)
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MockSearchInput(BaseModel):
    query: str = Field(description="Search query string")

class MockWebSearchTool(BaseTool):
    name: str = "Mock Web Search Tool"
    description: str = "Search the web for technical documentation and guides."
    args_schema: Type[BaseModel] = MockSearchInput

    def _run(self, query: str) -> str:
        return f"Results for '{query}': RAG connects LLMs to external vector databases like ChromaDB."

if __name__ == "__main__":
    tool = MockWebSearchTool()
    print(tool._run("fine-tuning vs RAG"))
