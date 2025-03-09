#pip install crewai crewai-tools
#pip install crewai sqlalchemy psycopg2-binary

import os
from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, start, listen
import sqlalchemy
from sqlalchemy import create_engine, text
import asyncio

# Set up environment variables (you'd typically use a .env file)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Database connection (PostgreSQL)
DB_URL = "postgresql+psycopg2://user:password@localhost:5432/sample_db"
engine = create_engine(DB_URL)

# Sample table schemas
TABLE_SCHEMAS = """
Table: customers
- id (integer, primary key)
- name (varchar)
- email (varchar)
- signup_date (date)

Table: orders
- order_id (integer, primary key)
- customer_id (integer, foreign key to customers.id)
- order_date (date)
- total_amount (decimal)
"""

# Sample previous questions and SQL queries
SAMPLE_QA = """
Question: How many customers signed up in 2024?
SQL: SELECT COUNT(*) as 2024_signups 
     FROM customers 
     WHERE signup_date >= '2024-01-01' AND signup_date < '2025-01-01';

Question: What is the total order amount per customer?
SQL: SELECT c.name, SUM(o.total_amount) as total_spent 
     FROM customers c 
     LEFT JOIN orders o ON c.id = o.customer_id 
     GROUP BY c.name;
"""

# Define Agents
schema_analyst = Agent(
    role="Schema Analyst",
    goal="Understand database schemas and provide context",
    backstory="Expert in database structures and relationships",
    verbose=True
)

sql_generator = Agent(
    role="SQL Generator",
    goal="Convert natural language questions to accurate SQL queries",
    backstory="Specialist in SQL query generation with 10+ years experience",
    verbose=True
)

query_executor = Agent(
    role="Query Executor",
    goal="Execute SQL queries and return results",
    backstory="Database administrator skilled in query optimization",
    verbose=True
)

# Define the Flow class for Text-to-SQL workflow
class TextToSQLFlow(Flow):
    def __init__(self):
        super().__init__()
        self.user_question = None
        
    # Starting point: Analyze schema and prepare context
    @start()
    async def analyze_schema(self):
        """Prepare schema context and sample Q&A"""
        self.user_question = "Which customers placed orders worth more than $1000 in 2025?"
        
        schema_task = Task(
            description=f"Analyze these schemas and sample Q&A to provide context:\nSchemas:\n{TABLE_SCHEMAS}\nSample Q&A:\n{SAMPLE_QA}",
            agent=schema_analyst,
            expected_output="A summary of schema relationships and relevant sample patterns"
        )
        
        crew = Crew(
            agents=[schema_analyst],
            tasks=[schema_task],
            verbose=2
        )
        
        result = await crew.kickoff_async()
        self.state["schema_context"] = result
        return result

    # Listen to schema analysis and generate SQL
    @listen("analyze_schema")
    async def generate_sql(self):
        """Generate SQL query based on user question and schema context"""
        schema_context = self.state["schema_context"]
        
        sql_task = Task(
            description=f"Generate a SQL query for this question: '{self.user_question}'\nUsing schema context: {schema_context}\nSchemas: {TABLE_SCHEMAS}\nSample Q&A: {SAMPLE_QA}",
            agent=sql_generator,
            expected_output="A valid SQL query string"
        )
        
        crew = Crew(
            agents=[sql_generator],
            tasks=[sql_task],
            verbose=2
        )
        
        result = await crew.kickoff_async()
        self.state["sql_query"] = result
        return result

    # Listen to SQL generation and execute query
    @listen("generate_sql")
    async def execute_query(self):
        """Execute the generated SQL query against PostgreSQL"""
        sql_query = self.state["sql_query"]
        
        execution_task = Task(
            description=f"Execute this SQL query against a PostgreSQL database and return results:\n{sql_query}",
            agent=query_executor,
            expected_output="Query results in a readable format"
        )
        
        # Custom execution logic within the agent simulation
        try:
            with engine.connect() as connection:
                result = connection.execute(text(sql_query))
                rows = result.fetchall()
                formatted_results = "\n".join([str(row) for row in rows])
                self.state["query_results"] = formatted_results
        except Exception as e:
            self.state["query_results"] = f"Error executing query: {str(e)}"
        
        return self.state["query_results"]

# Run the flow
async def main():
    flow = TextToSQLFlow()
    await flow.run()
    
    # Print the results
    print("\nUser Question:", flow.user_question)
    print("\nGenerated SQL Query:")
    print(flow.state.get("sql_query", "No query generated"))
    print("\nQuery Results:")
    print(flow.state.get("query_results", "No results available"))

if __name__ == "__main__":
    asyncio.run(main())
