"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import argparse
import asyncio
import dataclasses
import pdb
from enum import Enum, auto
import json
import logging
import time
from typing import List, Union
import threading
from pathlib import Path
# FastAPI framework, high performance, easy to learn, fast to code, ready for production
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from utils import build_logger, server_error_msg, CONTROLLER_HEART_BEAT_EXPIRATION
logger = build_logger("controller", "controller.log")

class DispatchMethod(Enum):
    LOTTERY = auto()  # we can get the assigned integer value automatically
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        # shortest_queue
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")

# 创建数据模型
# 当一个模型属性具有默认值时，它不是必需的。否则它是一个必需属性。将默认值设为 None 可使其成为可选属性
@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str


def heart_beat_controller(controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo] shortest_queue
        self.worker_info = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,))
        self.heart_beat_thread.start()

        logger.info("Init controller")

    def register_worker(self, worker_name: str, check_heart_beat: bool, worker_status: dict):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"],
            worker_status["speed"],
            worker_status["queue_length"],
            check_heart_beat,
            time.time()
        )

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)  # None
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None
        print('get_worker_status: ', r.json())
        # pdb.set_trace()
        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        return list(model_names)

    def get_worker_address(self, model_name: str):
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)),
                    p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # # Check status before returning
            # while True:
            #     pt = np.random.choice(np.arange(len(worker_names)),
            #         p=worker_speeds)
            #     worker_name = worker_names[pt]
            #
            #     if self.get_worker_status(worker_name):
            #         break
            #     else:
            #         self.remove_worker(worker_name)
            #         worker_speeds[pt] = 0
            #         norm = np.sum(worker_speeds)
            #         if norm < 1e-4:
            #             return ""
            #         worker_speeds = worker_speeds / norm
            #         continue
            # return worker_name

        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            if len(worker_names) == 0:
                return ""
            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            self.worker_info[w_name].queue_length += 1
            logger.info(f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}")
            return w_name
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def worker_api_generate_stream(self, params):
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"

        # try:
            # response = requests.post(worker_addr + "/worker_generate_stream", json=params, stream=True, timeout=15)
            # print('========= control response: ', response)
            # for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            #     print('========= control chunk: ', chunk)
            #     if chunk:
            #         yield chunk + b"\0"m
        # except requests.exceptions.RequestException as e:  # All exceptions that Requests explicitly raises inherit from e
        #     logger.info(f"worker timeout: {worker_addr}")
        #     ret = {
        #         "text": server_error_msg,
        #         "error_code": 3,
        #     }
        #     yield json.dumps(ret).encode() + b"\0"
        # todo: https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
        try:
            response = requests.post(worker_addr + "/worker_generate_stream", json=params, stream=True, timeout=15)
            print('========= control response: ', response)
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                print('========= control chunk: ', chunk)
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print('-------bad---------')
            SystemExit(e)

    # Let the controller act as a worker to achieve hierarchical management.
    # This can be used to connect isolated sub networks.
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
        }


app = FastAPI()

@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    re = controller.register_worker(
        data["worker_name"],
        data["check_heart_beat"],
        data.get("worker_status", None)
    )
    print('data: ', data)
    print('re: ', re)


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    models = controller.refresh_all_workers()


@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(
        data["worker_name"], data["queue_length"])
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    params = await request.json()
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)

# https://fastapi.tiangolo.com/zh/tutorial/body/
# 声明请求体
# 定义 path operation 为 POST，命名路径为 /worker_get_status，
@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    return controller.worker_api_get_status()  # r = requests.post(worker_name + "/worker_get_status", timeout=5)


# http://0.0.0.0:21001/docs

if __name__ == "__main__":
    '''
    python3 -m controller --host 0.0.0.0 --port 21001
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--dispatch-method", type=str, choices=["lottery", "shortest_queue"], default="shortest_queue")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller = Controller(args.dispatch_method)
    # Uvicorn is an ASGI web server implementation for Python.
    # 启动 ASGI 应用程序。ASGI (Asynchronous Server Gateway Interface) 是一种服务器接口，用于在 Web 服务器和 Web 应用程序之间传递请求和响应。
    # app 是 FastAPI 应用程序的变量名，FastAPI 是一个用于构建 Web API 的 Python 库。它使用了 asyncio 和 Starlette 等底层库来提供高性能的 API 开发体验。
    # host 是你希望应用程序侦听的主机的 IP 地址。 0.0.0.0 表示应用程序应侦听所有可用的 IP 地址。
    # port 是你希望应用程序侦听的端口号。这里的 21001 表示应用程序应侦听端口 21001。
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=args.reload)
    # 因此，这一行代码的意思是：启动 FastAPI 应用程序，并侦听所有可用的 IP 地址的端口 8000。

    # uvicorn.run(f"{Path(__file__).stem}:app", host=args.host, port=args.port, log_level="info", reload=args.reload)
    # uvicorn.run("__main__:app", host=args.host, port=args.port, log_level="info", reload=args.reload)
    # Please note that in this case, if you use reload=True or workers=NUM, you should put uvicorn.run into if __name__ == '__main__' clause in the main module.


    '''
    0.0.0.0 means "listen on all interfaces present on this computer"
    python3 -m controller --host 0.0.0.0 --port 21001
    python3 -m controller --host 127.0.0.1
    python3 -m controller --host localhost
    '''