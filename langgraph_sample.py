import langgraph
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate

# Initialize LangGraph
workflow = langgraph.Workflow()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Function to analyze user input and extract details
def analyze_input():
    user_input = input("Enter actor search query: ")
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Extract the actor's name, guild, ACTNUM number, or internal ID from the following user query.
        Determine the intent and which API should be called based on the query.
        Query: {query}
        Respond in JSON format: {{'intent': 'api_name', 'name': '...', 'guild': '...', 'actnum_number': '...', 'internal_id': '...'}}
        """
    )
    response = llm.predict(prompt.format(query=user_input))
    return eval(response)  # Convert JSON string to dict

# Function to call the smart search API
def call_smart_search(actor_info):
    print("Calling smart search API with:", actor_info)
    # Mock API response
    matches = [
        {"name": "Mr. John Doe", "internal_number": "internal123", "actnum": "1234567890"},
        {"name": "Mr. Jane Smith", "internal_number": "internal456", "actnum": "0987654321"}
    ]
    return matches

# Function to get user selection if multiple matches are found
def ask_for_specific_match(matches):
    print("Multiple matches found:")
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match['name']} - internal: {match['internal_number']}, ACTNUM: {match['actnum']}")
    choice = input("Enter the Internal number or ACTNUM number of the correct actor: ")
    selected_match = next((m for m in matches if m['internal_number'] == choice or m['actnum'] == choice), None)
    return selected_match

# Function to call the final API for a specific actor
def call_final_api(actor):
    print("Calling final API for:", actor)
    return {"status": "Success", "actor": actor}

# Graph Nodes
workflow.add_node("analyze_input", analyze_input)
workflow.add_node("search_api", call_smart_search)
workflow.add_node("resolve_match", ask_for_specific_match)
workflow.add_node("final_call", call_final_api)

# Define workflow logic
workflow.set_entry_point("analyze_input")
workflow.add_edge("analyze_input", "search_api")
workflow.add_conditional_edges(
    "search_api",
    lambda matches: "resolve_match" if len(matches) > 1 else "final_call"
)
workflow.add_edge("resolve_match", "final_call")

# Run the workflow
if __name__ == "__main__":
    result = workflow.run()
    print("Final result:", result)
