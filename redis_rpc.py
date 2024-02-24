import json
from functools import cache

import redis
from pydantic import BaseModel

REDIS_HOST="localhost"
REDIS_PORT="63679"


class LabelRequestMessage(BaseModel):
    id: str
    prompt: str
    image_paths: list[str]


class LabelResponseMessage(BaseModel):
    id: str
    label: str


@cache
def get_redis_conn():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT)


def queue_label_request(job: LabelRequestMessage):
    conn = get_redis_conn()
    conn.lpush("label_rpc::v2", job.json())


def block_wait_job() -> LabelRequestMessage:
    conn = get_redis_conn()
    _, encoded_data = conn.blpop(["label_rpc::v2"])
    return LabelRequestMessage(**json.loads(encoded_data.decode()))


def queue_job_response(response: LabelResponseMessage):
    conn = get_redis_conn()
    conn.lpush(f"label_rpc::v2::{response.id}", response.json())


def block_wait_response(id: str) -> LabelResponseMessage:
    conn = get_redis_conn()
    _, encoded_data = conn.blpop([f"label_rpc::v2::{id}"])
    return LabelResponseMessage(**json.loads(encoded_data.decode()))
