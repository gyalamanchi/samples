import chromadb
from sentence_transformers import SentenceTransformer
import yaml
import os

class VectorStore:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config['vector_store']['chroma_path']
        )
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup collections
        self.api_collection = self.chroma_client.get_or_create_collection(
            name=self.config['vector_store']['api_collection']
        )
        self.schema_collection = self.chroma_client.get_or_create_collection(
            name=self.config['vector_store']['schema_collection']
        )
        
        # Initialize with sample data if collections are empty
        self._initialize_data()

    def _initialize_data(self):
        # Initialize API examples if collection is empty
        if self.api_collection.count() == 0:
            sample_apis = {
                "api1": {
                    "endpoint": "GET /api/v1/users?age_gt=30",
                    "description": "Get all users over 30 years old"
                },
                "api2": {
                    "endpoint": "GET /api/v1/customers?fields=name,email",
                    "description": "Get customer names and emails"
                }
            }
            for id, data in sample_apis.items():
                embedding = self.model.encode(data["description"]).tolist()
                self.api_collection.add(
                    ids=[id],
                    embeddings=[embedding],
                    metadatas=[{"endpoint": data["endpoint"], "description": data["description"]}],
                    documents=[data["description"]]
                )

        # Initialize API schemas if collection is empty
        if self.schema_collection.count() == 0:
            sample_schemas = {
                "users": {
                    "endpoint": "/api/v1/users",
                    "fields": ["id", "name", "age", "email"],
                    "description": "Users resource endpoint"
                },
                "customers": {
                    "endpoint": "/api/v1/customers",
                    "fields": ["id", "name", "email", "phone"],
                    "description": "Customers resource endpoint"
                }
            }
            for resource_name, schema in sample_schemas.items():
                schema_text = f"API {schema['endpoint']}: {', '.join(schema['fields'])}"
                embedding = self.model.encode(schema_text).tolist()
                self.schema_collection.add(
                    ids=[resource_name],
                    embeddings=[embedding],
                    metadatas=[{
                        "endpoint": schema["endpoint"],
                        "fields": schema["fields"],
                        "description": schema["description"]
                    }],
                    documents=[schema_text]
                )

    def get_similar_api(self, question, top_k=3):
        query_embedding = self.model.encode(question).tolist()
        results = self.api_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        similar_apis = []
        for i in range(len(results['ids'][0])):
            endpoint = results['metadatas'][0][i]['endpoint']
            similarity = results['distances'][0][i]
            similar_apis.append((endpoint, 1 - similarity))  # Convert distance to similarity
        return similar_apis

    def get_relevant_schema(self, question):
        query_embedding = self.model.encode(question).tolist()
        results = self.schema_collection.query(
            query_embeddings=[query_embedding],
            n_results=2  # Get top 2 most relevant schemas
        )
        
        schemas = {}
        for i in range(len(results['ids'][0])):
            resource_name = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            schemas[resource_name] = {
                "endpoint": metadata["endpoint"],
                "fields": metadata["fields"]
            }
        return schemas

    def add_api_example(self, endpoint, description):
        embedding = self.model.encode(description).tolist()
        id = f"api_{self.api_collection.count() + 1}"
        self.api_collection.add(
            ids=[id],
            embeddings=[embedding],
            metadatas=[{"endpoint": endpoint, "description": description}],
            documents=[description]
        )

    def add_schema(self, resource_name, endpoint, fields):
        schema_text = f"API {endpoint}: {', '.join(fields)}"
        embedding = self.model.encode(schema_text).tolist()
        self.schema_collection.add(
            ids=[resource_name],
            embeddings=[embedding],
            metadatas=[{
                "endpoint": endpoint,
                "fields": fields,
                "description": f"{resource_name} resource endpoint"
            }],
            documents=[schema_text]
        )
