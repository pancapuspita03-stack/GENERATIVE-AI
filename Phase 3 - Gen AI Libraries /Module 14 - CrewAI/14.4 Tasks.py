# 14.4 Tasks (Halaman 156-159)
from crewai import Task

def make_tasks(researcher, writer, reviewer):
    research_task = Task(
        description="""Research the current state of RAG (Retrieval-Augmented Generation) in 2024-2025.
Cover:
1. What RAG is and how it works
2. The main variants (naive RAG, advanced RAG, modular RAG)
3. Common production challenges and how practitioners solve them
4. Three real-world use cases with specific results
Be specific and factual. Avoid vague generalities.""",
        expected_output="""A structured research report with:
- A clear definition and architecture overview
- Descriptions of the three RAG variants with key differences
- Three specific production challenges with concrete solutions
- Three real-world use cases with measurable outcomes
Format: Markdown with headers.""",
        agent=researcher,
    )

    writing_task = Task(
        description="""Using the research report provided, write a developer guide titled
'Building Production RAG Systems'.
The guide should:
- Be written for senior developers with Python experience
- Include a conceptual overview, implementation considerations, and common pitfalls
- Use active voice and concrete examples
- Be 600-800 words""",
        expected_output="A complete, publication-ready developer guide in Markdown format.",
        agent=writer,
        context=[research_task], # this task depends on research_task output
    )

    review_task = Task(
        description="""Review the developer guide for:
1. Technical accuracy - flag any claims that are incorrect or unsupported
2. Completeness - identify important topics that are missing
3. Clarity - suggest any sections that need rewriting
4. Audience fit - check that the level is appropriate for senior developers
Provide a structured review with specific, actionable feedback.""",
        expected_output="""A quality review report containing:
- Overall assessment (Ready / Needs Revision)
- Specific accuracy issues (if any) with corrections
- Missing content recommendations
- Clarity improvement suggestions
- Final verdict""",
        agent=reviewer,
        context=[writing_task],
    )
    return research_task, writing_task, review_task

if __name__ == "__main__":
    print("Task generator factory compiled.")
