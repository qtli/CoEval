"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import pdb
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
# from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch
import uvicorn

from utils import WORKER_HEART_BEAT_INTERVAL
from cli import load_openai_model, load_model, generate_stream
from utils import (build_logger, server_error_msg, pretty_print_semaphore)

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]  # c60530
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

class ModelWorker:
    def __init__(self,
                 controller_addr,
                 worker_addr,
                 worker_id,
                 no_register,
                 model_path,
                 model_name,
                 device,
                 num_gpus,
                 load_8bit=False,
                 args=None):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = model_name or model_path.split("/")[-1]
        self.device = device
        self.args = args
        logger.info(f"We will use the model {self.model_name} on worker {worker_id} ...")

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + len(model_semaphore._waiters)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def generate_stream_gate(self, params):
        try:
            for output in load_openai_model(
                params=params,
                args=self.args,
            ):
                ret = {
                    "text": output,
                    "error_code": 0,
                }
                # 返回一个可以用来迭代（for循环）的生成器
                yield json.dumps(ret).encode() + b"\0"
        except requests.exceptions.RequestException as e:
            ret = {
                "text": server_error_msg,
                "error_code": 3,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = worker.generate_stream_gate(params)
    # print('generator: ', generator)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    # Takes an async generator or a normal generator/iterator and streams the response body.
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")

    parser.add_argument("--task", type=str, default='eli5')

    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--stop_token", type=str, default='</s>', help="Token at which text generation is stopped")
    parser.add_argument("--temperature", type=float, default=0, help="temperature of 1.0 has no effect, lower tend toward greedy sampling. Setting temperature to 0 will make the outputs mostly deterministic",)
    parser.add_argument("--repetition_penalty", type=float, default=0, help="HF: primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--presence_penalty", type=float, default=0, help="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")
    parser.add_argument("--frequency_penalty", type=float, default=0, help="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,  # http://localhost:21002
                         args.worker_address,  # http://localhost:21001
                         worker_id,  # c60530
                         args.no_register,  # false
                         args.model_path,  # /path/to/weights
                         args.model_name,  # none
                         args.device,  # cuda
                         args.num_gpus,  # 1
                         args.load_8bit,
                         args=args)  # false
    # host: localhost, port: 21002
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    '''
    python3 -m model_worker --model-path gpt-3.5-turbo-0301 --model-name gpt-3.5-turbo-0301
    python3 -m model_worker --model-path gpt-3.5-turbo-0301 --model-name gpt-3.5-turbo-0301 --worker-address http://127.0.0.1:21002 --controller-address http://127.0.0.1:21001 --host 127.0.0.1
    python3 -m model_worker --model-path gpt-3.5-turbo-0301 --model-name gpt-3.5-turbo-0301 --host 0.0.0.0 --worker-address http://0.0.0.0:21002 --controller-address http://0.0.0.0:21001
    '''