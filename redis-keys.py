import redis
import config

redis_client = redis.Redis(host=config.redis_host, port=config.redis_port, password=config.redis_password, decode_responses=True)
keys = sorted(redis_client.keys(pattern=f"{{{config.redis_key}}}:*"))
print(keys)
