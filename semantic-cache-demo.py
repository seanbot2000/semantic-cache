import csv
import json
import time
import logging
import sys
import config
import numpy
from tqdm import tqdm
from argparse import ArgumentParser
from azure.core.exceptions import AzureError as exceptions
from azure.cosmos import CosmosClient, PartitionKey
import openai
import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

cosmos_endpoint = config.endpoint
cosmos_key = config.master_key
cosmos_database_id = config.database_id
cosmos_container_id = config.container_id
cosmos_partition_key = config.partition_key
data_file = config.data_file
cosmos_client = None
cosmos_db = None
cosmos_container = None

redis_client = None

jsonData = {}
vectorData = {}

def import_csv():
	logging.info("Import CSV file")
	with open(data_file, encoding='utf-8') as csvFile:	
		reader = csv.DictReader(csvFile)
		for index, row in enumerate(reader, start=1):
			jsonData[index] = row
	logging.info("CSV imported, JSON created")

def push_cosmos_data():
	logging.info("Connecting to Cosmos")
	cosmos_client = CosmosClient(url=cosmos_endpoint, credential=cosmos_key)
	logging.info("Create db if not exists")
	cosmos_db = cosmos_client.create_database_if_not_exists(id=cosmos_database_id)
	logging.info("Create container")
	cosmos_container = cosmos_db.create_container_if_not_exists(id=cosmos_container_id, partition_key=PartitionKey(path=cosmos_partition_key, kind='Hash'))

	logging.warning("Create Items")
	for key, value in jsonData.items():
		cosmos_container.create_item(body=value)

def create_vector_data():
	logging.info("configure connection to Azure Open AI")
	openai.api_type = "azure"
	openai.api_key = config.azure_openai_api_key
	openai.api_base = config.azure_openai_endpoint
	openai.api_version = config.azure_openai_api_version
	azure_openai_embedding_model = config.azure_openai_embedding_model
	logging.info("create vectors from data")
	for key, value in tqdm(jsonData.items()):
		vectorText = f"{value['Title']}, {value['Director']}, {value['Writer']}, {value['Summary']}"
		response = openai.Embedding.create(
    		input=vectorText,
    		engine=azure_openai_embedding_model
		)
		vectorData[key] = response['data'][0]['embedding']
		if key%50==0:
			time.sleep(10)

def create_redis_indexes(vectorDim):
	logging.info("create index with vector data")
	redis_client = redis.Redis(host=config.redis_host, port=config.redis_port, password=config.redis_password, decode_responses=True)
	schema = (
		TextField("$.Title", no_stem=True, as_name="title"),
		TextField("$.Director", no_stem=True, as_name="director"),
		NumericField("$.Year", as_name="year"),
		TagField("$.MainGenres", as_name="genres"),
		TextField("$.Summary", as_name="summary"),
		VectorField(
			"$.description_embeddings",
			"HNSW",
			{
				"TYPE": "FLOAT32",
				"DIM": 1536,
				"DISTANCE_METRIC": "COSINE",
			},
			as_name="vector",
		),
	)
	definition = IndexDefinition(prefix=[f"{{{config.redis_key}}}:"], index_type=IndexType.JSON)
	res = redis_client.ft(f"idx:{config.redis_key}").create_index(
		fields=schema, definition=definition
	)

	logging.info("second schema for q&a cache")
	schema = (
		NumericField("$.id", as_name="id"),
		TextField("$.Question", no_stem=True, as_name="question"),
		TextField("$.Title", no_stem=True, as_name="title"),
		TextField("$.Director", no_stem=True, as_name="director"),
		NumericField("$.Year", as_name="year"),
		TagField("$.MainGenres", as_name="genres"),
		TextField("$.Summary", as_name="summary"),
		VectorField(
			"$.description_embeddings",
			"HNSW",
			{
				"TYPE": "FLOAT32",
				"DIM": vectorDim,
				"DISTANCE_METRIC": "COSINE",
			},
			as_name="vector",
		),
	)
	definition = IndexDefinition(prefix=[f"{config.redis_key}_qa:"], index_type=IndexType.JSON)
	res = redis_client.ft(f"idx:{config.redis_key}_qa").create_index(
		fields=schema, definition=definition
	)


def push_redis_data():
	redis_client = redis.Redis(host=config.redis_host, port=config.redis_port, password=config.redis_password, decode_responses=True)
	logging.info("push JSON data to Redis for cache hits")
	pipeline = redis_client.pipeline()
	for key, value in tqdm(jsonData.items()):
		redis_key = f"{{{config.redis_key}}}:{key:03}"
		pipeline.json().set(redis_key, "$", value)
		if key%50==0:
			pipeline.execute()

	logging.info("iterate over keys so we can add embeddings")
	keys = sorted(redis_client.keys(pattern=f"{{{config.redis_key}}}:*"))
	logging.info("add embeddings to key")
	
	pipeline = redis_client.pipeline()
	index = 0
	for key in keys:
		index+=1
		pipeline.json().set(f"{key}", "$.description_embeddings", vectorData[index])
		if index%10==0:
			pipeline.execute()

	vectorDim = len(vectorData[1])
	create_redis_indexes(vectorDim)

def check_redis_for_query_match(query):
	logging.info("Redis semantic check against answered questions")
	redis_client = redis.Redis(host=config.redis_host, port=config.redis_port, password=config.redis_password, decode_responses=True)
	response = redis_client.ft(f"idx:{config.redis_key}_qa").search(f"@question:{query}")
	return response

def vss_search_redis(vectors):
	logging.info("running vss search on index in Redis")
	query = (
		Query('(*)=>[KNN 3 @vector $query_vector]')
		.return_fields('id', 'title', 'director')
		.dialect(2)
	)

	redis_client = redis.Redis(host=config.redis_host, port=config.redis_port, password=config.redis_password, decode_responses=True)
	result_docs = (
            redis_client.ft(f"idx:{config.redis_key}")
            .search(
                query, query_params=
                {
                    "query_vector": numpy.array(
                        vectors, dtype=numpy.float32
                    ).tobytes()
                },
            )
            .docs
        )
	print(result_docs)
	return result_docs
	
def query_cosmos(vssResults):
	logging.info("getting result from cosmosdb")
	cosmos_client = CosmosClient(url=cosmos_endpoint, credential=cosmos_key)
	cosmos_db = cosmos_client.get_database_client(cosmos_database_id)
	cosmos_db.read()
	cosmos_container = cosmos_db.get_container_client(cosmos_container_id)
	cosmos_container.read()

	responses = []
	for result in vssResults:
		print(result)
		responses.append(cosmos_container.read_item(item=vssResults["id"], partition_key=result["id"]))
		exit

	return responses

def cache_question_response_redis(query, cosmosResult):
	logging.info("caching query and response to Redis")
	print(cosmosResult)
	redis_client = redis.Redis(host=config.redis_host, port=config.redis_port, password=config.redis_password, decode_responses=True)
	redis_key = f"{{{config.redis_key}}}:{cosmosResult.id:03}"
	redis_client.json().set(redis_key, "$", cosmosResult)
	redis_client.json().set(redis_key, "$.Question", query)

def format_json_response(response):
	logging.info("formatting JSON response")
	formatted = None
	dict = None
	print(response)
	for item in response:
		dict = json.loads(item)
		formatted += f"Title: {item.title}, Director: {item.director}\nSummary: {item.summary}\n\n"

	return formatted

def interact():
	openai.api_type = "azure"
	openai.api_key = config.azure_openai_api_key
	openai.api_base = config.azure_openai_endpoint
	openai.api_version = config.azure_openai_api_version
	azure_openai_embedding_model = config.azure_openai_embedding_model

	queries = [
		"Napoleon",
		"Movies about cars",
		"Movies about cars and dogs",
		"Movies about school",
		"War movies",
	]

	logging.info("iterate over queries and see if there is a cache hit")
	for query in queries:
		print("checking to see if we have a match for: " + query)
		response = check_redis_for_query_match(query)
		if response.total != 0:
			logging.info("send cache hit message to user and continue")
			print("Cache hit: here is the response from Redis")
			formattedResponse = format_json_response(response.documents)
			print(formattedResponse)
			input("Press Enter to continue...")
		else:
			logging.info("miss for this question: create vector to get results")
			response = openai.Embedding.create(
				input=query,
				engine=azure_openai_embedding_model
			)
			vectors = response['data'][0]['embedding']
			logging.info("perform vector similarity search")
			vssResults = vss_search_redis(vectors)
			logging.info("get data from cosmos")
			cosmosResponse = query_cosmos(vssResults)
			print("Cosmos hit: here is the response from Cosmos")
			formattedResult = format_json_response(cosmosResponse)
			print(formattedResult)
			logging.info("Cache question and response data")
			cache_question_response_redis(query, cosmosResponse)
			input("Press Enter to continue...")

def set_logging(verbose):
	if verbose:
		logging.basicConfig(
			stream=sys.stdout, level=logging.INFO
		)
	else:
		logging.basicConfig(
			stream=sys.stdout, level=logging.ERROR
		)

def data_import():
	logging.info("Running data import.")
	import_csv()
	logging.info("push data to cosmos")
	#push_cosmos_data()
	logging.info("create vectors")
	create_vector_data()
	logging.info("push vectors to redis")
	push_redis_data()

def sematic_cache_demo():
	logging.info("ask questions and interact with data")
	interact()
	logging.info("Finished!")
	

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-i", "--import-data", 
					    action="store_true", dest="data", default=False,
						help="import data from file into cosmos and redis")
	parser.add_argument("-q", "--quiet",
						action="store_false", dest="verbose", default=True,
						help="don't print status messages to stdout")
	args = parser.parse_args()
	set_logging(args.verbose)
	if args.data:
		data_import()
	sematic_cache_demo()