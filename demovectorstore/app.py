
vector_store = VectorStore("config.yaml")
# Get similar API endpoints
similar_apis = vector_store.get_similar_api("find users older than 30")
# Returns something like: [("GET /api/v1/users?age_gt=30", 0.95), ...]

# Get relevant schemas
schemas = vector_store.get_relevant_schema("user information")
# Returns something like: {"users": {"endpoint": "/api/v1/users", "fields": ["id", "name", "age", "email"]}}

# Add new API endpoint
vector_store.add_api_example("GET /api/v1/users/{id}", "Get specific user by ID")

# Add new schema
vector_store.add_schema("profile", "/api/v1/profiles", ["id", "user_id", "bio"])
