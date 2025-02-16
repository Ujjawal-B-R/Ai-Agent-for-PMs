import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool, FileWriterTool
from dotenv import load_dotenv

load_dotenv()

readtool = FileReadTool()
openaillm = LLM(
  model = "ollama/mistral", 
  base_url="http://localhost:11434", 
  api_key="NA"
)

just_reader_agent = Agent(
    role = "Just a reader",
    goal = "Read the file in the given path {file_path}",
    backstory="You are expert in reading files",
    tools=[readtool],
    verbose=True,
    llm = openaillm,
)

product_manager_agent = Agent(
    role = "Product Manager",
    goal = "Create user stories and personas for the details provided in the file in the given path {file_path}",
    backstory="You're an expert product manager with a knack for crafting well structured and formatted user stories. You have a deep understanding of the product and it is your responsibility to ensure that the user stories and user personas you create are clear, actionable, and prioritized correctly.",
    verbose=True,
    llm = openaillm,
)

reader_task = Task(
    description="Read the file in the given path and return the content",
    expected_output="Get the number of words in the file",
    agent= just_reader_agent,
)

persona_creation_task = Task(
    description="Create comprehensive user personas and user stories based on the content of the file. The user personas should be well structured and formatted. The user personas should contain details like pain points, frustrations, goals, and motivations of the users.",
    expected_output="A well formatted and well structured report with user personas.",
    agent=product_manager_agent,
    prerequisites=[reader_task],
    context=[reader_task],
    output_file="user_story_creator_v2/UserPersonaReport.txt",
    )

reader_crew = Crew(
    agents = [just_reader_agent, product_manager_agent],
    tasks = [reader_task, persona_creation_task],
    Process = Process.sequential,
    Verbose = True,
)
reader_crew.kickoff({"file_path": r"read the file at C:\Users\Asus\crewAI-examples\starter_template\user_story_creator_v2\Filereader_v1.txt"})
