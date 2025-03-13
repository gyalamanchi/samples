#pip install psycopg2-binary langchain redis sentence-transformers

import psycopg2
from langchain.vectorstores import Redis
from langchain.embeddings import SentenceTransformerEmbeddings
from typing import List, Dict
import json

class TableVectorStore:
    def __init__(self, db_config: Dict, redis_url: str, schema: str = "public"):
        """Initialize with database and Redis configuration"""
        self.db_config = db_config
        self.redis_url = redis_url
        self.schema = schema
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.conn = None
        self.cursor = None

    def _connect_db(self):
        """Establish PostgreSQL connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()

    def _close_db(self):
        """Close PostgreSQL connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn and not self.conn.closed:
            self.conn.close()

    def extract_tables(self) -> List[Dict]:
        """Extract table metadata from PostgreSQL"""
        self._connect_db()
        
        columns_query = """
        SELECT 
            t.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            CASE WHEN pk.constraint_name IS NOT NULL THEN 'YES' ELSE 'NO' END as is_primary_key
        FROM information_schema.tables t
        LEFT JOIN information_schema.columns c 
            ON t.table_name = c.table_name 
            AND t.table_schema = c.table_schema
        LEFT JOIN (
            SELECT tc.table_name, c.column_name, tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = %s
        ) pk ON pk.table_name = t.table_name AND pk.column_name = c.column_name
        WHERE t.table_schema = %s
        ORDER BY t.table_name, c.column_name
        """

        fk_query = """
        SELECT 
            tc.table_name,
            kcu.column_name,
            ccu.table_name AS referenced_table,
            ccu.column_name AS referenced_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = %s
        """

        self.cursor.execute(columns_query, (self.schema, self.schema))
        columns_result = self.cursor.fetchall()
        
        self.cursor.execute(fk_query, (self.schema,))
        fk_result = self.cursor.fetchall()

        tables = {}
        for table_name, column_name, data_type, is_nullable, is_pk in columns_result:
            if table_name not in tables:
                tables[table_name] = {
                    'columns': [],
                    'foreign_keys': []
                }
            if column_name:
                tables[table_name]['columns'].append({
                    'name': column_name,
                    'type': data_type,
                    'nullable': is_nullable == 'YES',
                    'primary_key': is_pk == 'YES'
                })

        for table_name, column_name, ref_table, ref_column in fk_result:
            if table_name in tables:
                tables[table_name]['foreign_keys'].append({
                    'column': column_name,
                    'references': {
                        'table': ref_table,
                        'column': ref_column
                    }
                })

        self._close_db()
        return [{'table_name': k, **v} for k, v in tables.items()]

    def create_vector_store(self):
        """Create and populate Redis vector store"""
        table_data = self.extract_tables()
        
        # Convert table metadata to texts and metadata
        texts = []
        metadatas = []
        for table in table_data:
            content = f"Table: {table['table_name']}\n"
            content += "Columns:\n" + "\n".join(
                [f"- {col['name']} ({col['type']}, nullable: {col['nullable']}, pk: {col['primary_key']})" 
                 for col in table['columns']]
            )
            if table['foreign_keys']:
                content += "\nForeign Keys:\n" + "\n".join(
                    [f"- {fk['column']} -> {fk['references']['table']}.{fk['references']['column']}" 
                     for fk in table['foreign_keys']]
                )
            
            texts.append(content)
            metadatas.append({'table_name': table['table_name']})

        # Create Redis vector store
        self.vector_store = Redis.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            redis_url=self.redis_url,
            index_name='table_index'
        )

    def search_tables(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar tables based on query"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [{
            'table_name': doc.metadata['table_name'],
            'content': doc.page_content,
            'score': score
        } for doc, score in results]

def main():
    # Configuration
    db_config = {
        'dbname': 'your_database',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'localhost',
        'port': '5432'
    }
    
    redis_url = "redis://localhost:6379"
    schema = "public"

    # Create and use the vector store
    table_store = TableVectorStore(db_config, redis_url, schema)
    table_store.create_vector_store()
    
    # Example search
    query = "tables with user information and foreign keys"
    results = table_store.search_tables(query)
    
    # Print results
    for result in results:
        print(f"\nTable: {result['table_name']} (Score: {result['score']})")
        print(result['content'])
        print("-" * 50)

if __name__ == "__main__":
    main()

#USAGE
# Initialize
#table_store = TableVectorStore(db_config, redis_url, schema)

# Get raw table data if needed
#tables = table_store.extract_tables()

# Create vector store
#table_store.create_vector_store()

# Search
#results = table_store.search_tables("tables related to customers and orders")
#for result in results:
#    print(result['table_name'], result['score'])
#    print(result['content'])
