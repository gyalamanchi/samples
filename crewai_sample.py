from crewai import Crew, Agent, Task
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Define Agents
analyzer = Agent(
    name="Analyzer",
    role="Extracts actor information and determines API intent",
    backstory="This agent takes a user's input and extracts relevant actor details such as name, guild, actnum, or internal number.",
    llm=llm,
    function=lambda query: eval(
        llm.predict(
            PromptTemplate(
                input_variables=["query"],
                template="""
                Extract the actor's name, guild, actnum, or internal number from the following user query.
                Determine the intent and which API should be called based on the query.
                Query: {query}
                Respond in JSON format: {{'intent': 'api_name', 'name': '...', 'guild': '...', 'actnum': '...', 'internal_number': '...'}}
                """
            ).format(query=query)
        )
    )
)

search_agent = Agent(
    name="SearchAgent",
    role="Calls the smart search API",
    backstory="Queries the smart search API based on the extracted actor details.",
    function=lambda actor_info: [
        {"name": "John Doe", "internal_number": "IN123", "actnum": "1234567890"},
        {"name": "Jane Smith", "internal_number": "IN456", "actnum": "0987654321"}
    ]
)

resolver = Agent(
    name="Resolver",
    role="Handles multiple search results and asks the user for a specific match",
    backstory="If multiple actors are found, prompts the user to select the correct one.",
    function=lambda matches: next((m for m in matches if input(f"Enter Internal Number or Actnum for {m['name']}: ") in (m['internal_number'], m['actnum'])), None)
)

api_caller = Agent(
    name="APICaller",
    role="Calls the final API for the selected actor",
    backstory="Takes the resolved actor information and makes the final API call.",
    function=lambda actor: {"status": "Success", "actor": actor}
)

# Define Crew and Tasks
crew = Crew(
    agents=[analyzer, search_agent, resolver, api_caller],
    tasks=[
        Task(description="Analyze input query", agent=analyzer),
        Task(description="Call smart search API", agent=search_agent),
        Task(description="Resolve multiple matches", agent=resolver),
        Task(description="Call final API", agent=api_caller)
    ]
)

# Run the workflow
if __name__ == "__main__":
    user_input = input("Enter actor search query: ")
    result = crew.kickoff(user_input)
    print("Final result:", result)
