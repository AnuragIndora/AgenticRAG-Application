from pymilvus import connections, utility, Collection

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# List all collections
collections = utility.list_collections()
print("Collections:", collections)

# Choose your collection
collection_name = "agentic_rag_chunks"

# Get collection stats
stats = utility.has_collection(collection_name)
print("Collection stats:", stats)

# Inspect schema
col = Collection(collection_name)
print("Collection schema:", col.schema)

# Number of entities
print("Number of entities:", col.num_entities)

# Query first few entities (metadata fields)
fields = [field.name for field in col.schema.fields if field.name != "embedding"]  # skip vector field
entities = col.query(expr="id >= 0", output_fields=fields)
print("First few entities:", entities[:5])