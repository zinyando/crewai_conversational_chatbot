from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
)


@CrewBase
class CrewaiConversationalChatbotCrew:
    """CrewaiConversationalChatbot crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def assistant(self) -> Agent:
        return Agent(
            config=self.agents_config["assistant"],
            llm=llm,
            verbose=False,
        )

    @task
    def assistant_task(self) -> Task:
        return Task(config=self.tasks_config["assistant_task"], agent=self.assistant())

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiConversationalChatbot crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=0,
        )
