# 14.6 Hierarchical Process with a Manager (Halaman 161-163)
from crewai import Crew, Process, Agent, Task
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

    manager = Agent(
        role="AI Content Production Manager",
        goal="Coordinate a team to produce high-quality AI content efficiently.",
        backstory="""You are an experienced content production manager who excels at
breaking down projects, assigning work to the right specialist, and ensuring
quality standards are met before publication.""",
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )
    researcher = Agent(
        role="AI Research Analyst",
        goal="Research AI topics thoroughly.",
        backstory="Senior researcher specialising in LLM systems and RAG.",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write developer-focused technical content.",
        backstory="Technical writer with Python and ML experience.",
        llm=llm,
        verbose=False,
    )

    main_task = Task(
        description="""Produce a complete technical brief on 'Vector Databases for RAG in Production'.
Include: what they are, how to choose one, and three implementation best practices.""",
        expected_output="A structured technical brief in Markdown, 400-500 words.",
        agent=manager,
    )

    crew = Crew(
        agents=[manager, researcher, writer],
        tasks=[main_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=False,
    )
    print("Hierarchical Manager-led Crew compiled successfully.")
