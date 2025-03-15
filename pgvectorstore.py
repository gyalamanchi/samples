#pip install psycopg2-binary pgvector langchain sentence-transformers

from typing import List, Optional, Any
import psycopg2
from psycopg2.extras import execute_values
from langchain.embeddings import SentenceTransformerEmbeddings
import logging

class PGVectorStore:
    def __init__(
        self,
        connection_string: str,
        model_name: str = "all-MiniLM-L6-v2",  # Default model
        table_name: str = "vector_store",
        schema: str = "public"
    ):
        """
        Initialize the PGVector store with SentenceTransformerEmbeddings.
        
        Args:
            connection_string: PostgreSQL connection string
            model_name: SentenceTransformer model name
            table_name: Name of the table to store vectors
            schema: Database schema to use
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.schema = schema
        self.logger = logging.getLogger(__name__)
        
        # Initialize SentenceTransformerEmbeddings from LangChain
        self.embeddings = SentenceTransformerEmbeddings(model_name)
        
        # Get embedding dimension (assuming first embedding gives us the dimension)
        test_embedding = self.embeddings.embed_query("test")
        self.embedding_dim = len(test_embedding)
        
        # Initialize the database table
        self._create_table()

    def _get_connection(self):
        """Create a database connection."""
        try:
            return psycopg2.connect(self.connection_string)
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def _create_table(self):
        """Create the vector table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema}.{self.table_name} (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding VECTOR({self.embedding_dim})
        );
        CREATE INDEX IF NOT EXISTS {self.table_name}_vector_idx 
        ON {self.schema}.{self.table_name} 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(create_table_sql)
                    conn.commit()
                    self.logger.info(f"Table {self.table_name} initialized successfully")
                except Exception as e:
                    conn.rollback()
                    self.logger.error(f"Failed to create table: {e}")
                    raise

    def add_vectors(
        self,
        contents: List[str],
        metadata: Optional[List[dict]] = None
    ) -> List[int]:
        """
        Add vectors to the store by encoding contents.
        
        Args:
            contents: List of text contents to encode
            metadata: Optional list of metadata dictionaries
            
        Returns:
            List of inserted IDs
        """
        # Generate embeddings using SentenceTransformerEmbeddings
        embeddings = self.embeddings.embed_documents(contents)
        
        if metadata and len(metadata) != len(contents):
            raise ValueError("Number of metadata entries must match number of contents")
        
        # Convert embeddings to pgvector format
        vector_strings = [str(vec) for vec in embeddings]
        metadata = metadata or [{}] * len(contents)
        
        insert_sql = f"""
        INSERT INTO {self.schema}.{self.table_name} (content, embedding, metadata)
        VALUES %s
        RETURNING id
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    data = list(zip(contents, vector_strings, metadata))
                    ids = execute_values(
                        cur,
                        insert_sql,
                        data,
                        template="(%s, %s::vector, %s::jsonb)",
                        fetch=True
                    )
                    conn.commit()
                    return [row[0] for row in ids]
                except Exception as e:
                    conn.rollback()
                    self.logger.error(f"Failed to insert vectors: {e}")
                    raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[dict]:
        """
        Perform a cosine similarity search with text query.
        
        Args:
            query: Text query to search for
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries containing id, content, metadata, and similarity
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = str(query_embedding)
        
        search_sql = f"""
        SELECT 
            id,
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM {self.schema}.{self.table_name}
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY similarity DESC
        LIMIT %s
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(search_sql, (query_vector, query_vector, threshold, k))
                    results = cur.fetchall()
                    return [
                        {
                            "id": row[0],
                            "content": row[1],
                            "metadata": row[2],
                            "similarity": row[3]
                        }
                        for row in results
                    ]
                except Exception as e:
                    self.logger.error(f"Failed to perform similarity search: {e}")
                    raise

    def delete_vectors(self, ids: List[int]):
        """Delete vectors by their IDs."""
        delete_sql = f"""
        DELETE FROM {self.schema}.{self.table_name}
        WHERE id = ANY(%s)
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(delete_sql, (ids,))
                    conn.commit()
                    self.logger.info(f"Deleted {len(ids)} vectors")
                except Exception as e:
                    conn.rollback()
                    self.logger.error(f"Failed to delete vectors: {e}")
                    raise

# Example usage:
if __name__ == "__main__":
    # Example connection string
    conn_string = "postgresql://user:password@localhost:5432/dbname"
    
    # Initialize the vector store
    vector_store = PGVectorStore(
        connection_string=conn_string,
        model_name="all-MiniLM-L6-v2",
        table_name="my_vectors"
    )
    
    # Example data
    contents = ["This is a test document", "Another test document"]
    metadata = [
        {"source": "doc1"},
        {"source": "doc2"}
    ]
    
    # Add vectors
    ids = vector_store.add_vectors(contents, metadata)
    print(f"Inserted IDs: {ids}")
    
    # Perform similarity search
    results = vector_store.similarity_search(
        query="test document",
        k=2,
        threshold=0.5
    )
    print(f"Search results: {results}")
