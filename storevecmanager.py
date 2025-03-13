#pip install psycopg2-binary langchain redis sentence-transformers
#pip install langchain-community

import psycopg2
from langchain.vectorstores import Redis
from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    """Database configuration parameters"""
    dbname: str
    user: str
    password: str
    schema: str  # Added schema parameter
    host: str = "localhost"
    port: str = "5432"
    redis_url: str = "redis://localhost:6379"

class SchemaManager:
    """Manages database schema extraction and vector storage"""
    
    def __init__(self, config: DatabaseConfig, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with configuration and optional SentenceTransformer model"""
        self._config = config
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._schemas: Optional[List[Dict]] = None
        self._vector_store: Optional[Redis] = None
        self._embedding_model_name = model_name
    
    def _connect(self) -> None:
        """Establish database connection if not already connected"""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(
                    dbname=self._config.dbname,
                    user=self._config.user,
                    password=self._config.password,
                    host=self._config.host,
                    port=self._config.port
                )
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Postgres: {e}")

    def _extract_schemas(self) -> List[Dict]:
        """Extract schema information from Postgres using specified schema"""
        self._connect()
        schemas = []
        
        with self._conn.cursor() as cur:
            # Parameterized query to prevent SQL injection
            cur.execute("""
                SELECT 
                    t.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default
                FROM information_schema.tables t
                JOIN information_schema.columns c 
                    ON t.table_name = c.table_name
                WHERE t.table_schema = %s
                ORDER BY t.table_name, c.column_name
            """, (self._config.schema,))
            
            current_table = None
            table_schema = None
            
            for row in cur.fetchall():
                table_name, column_name, data_type, is_nullable, column_default = row
                
                if current_table != table_name:
                    if table_schema is not None:
                        schemas.append(table_schema)
                    table_schema = {
                        "table_name": table_name,
                        "columns": []
                    }
                    current_table = table_name
                
                table_schema["columns"].append({
                    "column_name": column_name,
                    "data_type": data_type,
                    "is_nullable": is_nullable == "YES",
                    "default": column_default
                })
            
            if table_schema is not None:
                schemas.append(table_schema)
                
        return schemas

    def _schema_to_text(self, schema: Dict) -> str:
        """Convert schema dictionary to text representation"""
        text = f"Table: {schema['table_name']}\n"
        for col in schema["columns"]:
            text += f"- {col['column_name']} ({col['data_type']})"
            text += " NULLABLE" if col["is_nullable"] else " NOT NULL"
            if col["default"]:
                text += f" DEFAULT {col['default']}"
            text += "\n"
        return text

    @property
    def _schemas_(self) -> List[Dict]:
        """Lazy load and return schemas"""
        if self._schemas is None:
            self._schemas = self._extract_schemas()
        return self._schemas

    @property
    def _vector_store_(self) -> Redis:
        """Lazy load and return vector store"""
        if self._vector_store is None:
            embeddings = SentenceTransformerEmbeddings(model_name=self._embedding_model_name)
            documents = [self._schema_to_text(schema) for schema in self._schemas_]
            metadatas = [{"table_name": schema["table_name"]} for schema in self._schemas_]
            
            self._vector_store = Redis.from_texts(
                texts=documents,
                embedding=embeddings,
                metadatas=metadatas,
                redis_url=self._config.redis_url,
                index_name=f"{self._config.schema}_table_schemas"  # Include schema in index name
            )
        return self._vector_store

    def _search_schemas_(self, query: str, k: int = 3) -> List:
        """Search for similar schemas"""
        return self._vector_store_.similarity_search(query, k=k)

    def _refresh_schemas_(self) -> None:
        """Force refresh of schemas and vector store"""
        self._schemas = None
        self._vector_store = None
        # Trigger lazy loading
        _ = self._schemas_
        _ = self._vector_store_

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, '_conn') and self._conn is not None and not self._conn.closed:
            self._conn.close()

# Usage example
def main():
    config = DatabaseConfig(
        dbname="your_database",
        user="your_user",
        password="your_password",
        schema="public"  # Specify the schema name here
    )
    
    try:
        # Initialize manager with default model or specify another
        manager = SchemaManager(config)
        
        # Access schemas (loaded on demand)
        print(f"Found {len(manager._schemas_)} tables in schema '{config.schema}'")
        
        # Perform a search
        results = manager._search_schemas_("tables with nullable columns")
        print("\nSearch results:")
        for doc in results:
            print(f"Table: {doc.metadata['table_name']}")
            print(f"Content: {doc.page_content[:100]}...")
            print("---")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
