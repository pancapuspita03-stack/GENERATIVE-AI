# 14.5 Crew - Putting It Together (Halaman 159-161)
from crewai import Crew, Process, Agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from poin_14_3_tools.custom_agent_tools import cost_tool, context_tool
    from poin_14_4_tasks.dependent_tasks import make_tasks

    llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

    researcher = Agent(
        role="AI Research Analyst",
        goal="Find accurate information on AI topics.",
        backstory="Senior AI research analyst with expertise in LLM systems.",
        llm=llm,
        tools=[context_tool],
        verbose=False,
        max_iter=5,
    )
    writer = Agent(
        role="Technical Content Writer",
        goal="Write clear technical content for developers.",
        backstory="Technical writer with a software engineering background.",
        llm=llm,
        tools=[cost_tool],
        verbose=False,
        max_iter=5,
    )
    reviewer = Agent(
        role="Content Quality Reviewer",
        goal="Ensure content is accurate and publication-ready.",
        backstory="Meticulous editor who reviews technical AI content.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    print("Sequential Crew assembled successfully.")
