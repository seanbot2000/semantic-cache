import os

endpoint = os.environ.get("COSMOS_ENDPOINT")
master_key = os.environ.get("COSMOS_KEY")
database_id = os.environ.get("COSMOS_DATABASE_ID")
container_id = os.environ.get("COSMOS_CONTAINER_ID")
partition_key = os.environ.get("COSMOS_PARTITION_KEY")
data_file = os.environ.get("DATA_FILE")
azure_openai_api_key = os.environ.get("AZURE_OPENAI_KEY")
azure_openai_api_version = os.environ.get("AZURE_API_VERSION")
azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") 
azure_openai_embedding_model = os.environ.get("AZURE_EMBEDDING_MODEL")
redis_host = os.environ.get("REDIS_HOST")
redis_password = os.environ.get("REDIS_PASSWORD")
redis_port = os.environ.get("REDIS_PORT")
redis_user = os.environ.get("REDIS_USER")
redis_key = os.environ.get("REDIS_KEY")