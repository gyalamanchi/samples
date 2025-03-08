import langgraph.graph as lg
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import OpenAIEmbeddings
import psycopg2

class GraphState(TypedDict):
    messages: List[BaseMessage]
    sql_query: str
    db_config: Dict[str, Any]
    relevant_sqls: List[str]
    relevant_schemas: List[str]
    is_valid: bool
    user_input: str

def get_user_input(state: GraphState):
    user_input = state["messages"][-1].content
    return {"messages": state["messages"], "user_input": user_input, "db_config": state["db_config"]}

def get_relevant_sqls(state: GraphState):
    user_input = state["user_input"]
    db_config = state["db_config"]
    # Simulated SQL examples and their embeddings. In a real scenario, these would be loaded from a database or file.
    sql_examples = ["SELECT * FROM customers WHERE city = 'New York';", "SELECT order_id, total FROM orders WHERE order_date > '2023-01-01';", "SELECT product_name FROM products WHERE category = 'Electronics';"]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(sql_examples, embeddings)
    relevant_sqls = vectorstore.similarity_search(user_input, k=2)
    relevant_sqls_texts = [doc.page_content for doc in relevant_sqls]
    return {"messages": state["messages"], "user_input": user_input, "db_config": db_config, "relevant_sqls": relevant_sqls_texts}

def get_relevant_schemas(state: GraphState):
    user_input = state["user_input"]
    db_config = state["db_config"]

    # Simulate schema retrieval based on user input (replace with actual schema retrieval)
    schemas = {
        "customers": "CREATE TABLE customers (customer_id INT, name VARCHAR, city VARCHAR);",
        "orders": "CREATE TABLE orders (order_id INT, customer_id INT, order_date DATE, total DECIMAL);",
        "products": "CREATE TABLE products (product_id INT, product_name VARCHAR, category VARCHAR);"
    }

    relevant_schemas_texts = []
    for table_name, schema in schemas.items():
        if table_name in user_input.lower():
            relevant_schemas_texts.append(schema)

    return {"messages": state["messages"], "user_input": user_input, "db_config": db_config, "relevant_schemas": relevant_schemas_texts}

def generate_sql(state: GraphState):
    user_input = state["user_input"]
    db_config = state["db_config"]
    relevant_sqls = state["relevant_sqls"]
    relevant_schemas = state["relevant_schemas"]

    prompt = ChatPromptTemplate.from_template(
        """
        Given the user question, generate a SQL query to answer it.
        Relevant SQLs: {relevant_sqls}
        Relevant Schemas: {relevant_schemas}
        Question: {user_input}
        SQL:
        """
    )

    llm = ChatOpenAI(temperature=0)
    chain = prompt | llm | StrOutputParser()
    sql_query = chain.invoke(
        {
            "user_input": user_input,
            "relevant_sqls": "\n".join(relevant_sqls),
            "relevant_schemas": "\n".join(relevant_schemas),
        }
    )

    return {"messages": state["messages"], "sql_query": sql_query, "db_config": db_config, "user_input": user_input}

def validate_sql(state: GraphState):
    sql_query = state["sql_query"]
    db_config = state["db_config"]

    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute(f"EXPLAIN {sql_query}") #just explain, do not run.
        cur.close()
        conn.close()
        return {"messages": state["messages"], "sql_query": sql_query, "db_config": db_config, "is_valid": True, "user_input": state["user_input"]}
    except Exception as e:
        return {"messages": state["messages"] + [HumanMessage(content=f"SQL Validation Error: {e}")], "sql_query": sql_query, "db_config": db_config, "is_valid": False, "user_input": state["user_input"]}

def execute_sql(state: GraphState):
    sql_query = state["sql_query"]
    db_config = state["db_config"]

    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute(sql_query)
        results = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return {"messages": state["messages"] + [HumanMessage(content=f"Results: {results}, columns: {column_names}")], "db_config": state["db_config"], "sql_query":sql_query, "user_input": state["user_input"]}
    except Exception as e:
        return {"messages": state["messages"] + [HumanMessage(content=f"SQL Execution Error: {e}")], "db_config": state["db_config"], "sql_query":sql_query, "user_input": state["user_input"]}

def present_results(state: GraphState):
    return {"messages": state["messages"], "db_config": state["db_config"], "sql_query": state["sql_query"], "user_input": state["user_input"]}

def create_graph():
    builder = lg.GraphBuilder()
    builder.add_node("user_input", get_user_input)
    builder.add_node("relevant_sqls", get_relevant_sqls)
    builder.add_node("relevant_schemas", get_relevant_schemas)
    builder.add_node("generate_sql", generate_sql)
    builder.add_node("validate_sql", validate_sql)
    builder.add_node("execute_sql", execute_sql)
    builder.add_node("present_results", present_results)

    builder.add_edge("user_input", "relevant_sqls")
    builder.add_edge("relevant_sqls", "relevant_schemas")
    builder.add_edge("relevant_schemas", "generate_sql")
    builder.add_edge("generate_sql", "validate_sql")
    builder.add_conditional_edges(
        "validate_sql",
        lambda state: "execute_sql" if state["is_valid"] else "present_results",
        {
            "execute_sql": "execute_sql",
            "present_results": "present_results"
        }
    )

    builder.add_edge("execute_sql", "present_results")

    return builder.compile()

def run_graph(user_question: str, db_config: Dict[str, Any]):
    app = create_graph()
    state = {
        "messages": [HumanMessage(content=user_question)],
        "sql_query": "",
        "db_config": db_config,
        "relevant_sqls": [],
        "relevant_schemas": [],
        "is_valid": False,
        "user_input": ""
    }
    for output in app.stream(state):
        for key, value in output.items():
            if isinstance(value, dict) and "messages" in value:
                state.update(value)
    return state["messages"][-1].content

if __name__ == "__main__":
    db_config = {
        "host": "your_host",
        "port": "your_port",
        "database": "your_database",
        "user": "your_user",
        "password": "your_password",
    }

    user_question = "Get the names of customers from New York."
    result = run_graph(user_question, db_config)
    print(result)
