"""
Manually register workers.

Usage:
python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name http://localhost:21002

Python -m 指的是把这个.py文件当做模块运行
"""

import argparse
import pdb

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str)
    parser.add_argument("--worker-name", type=str)
    parser.add_argument("--check-heart-beat", action="store_true")
    args = parser.parse_args()

    url = args.controller_address + "/register_worker"  # http://localhost:21001/register_worker
    data = {
        "worker_name": args.worker_name,  # https://
        "check_heart_beat": args.check_heart_beat,  # false
        "worker_status": None,
    }
    # Make a POST request to a web page, and return the response text:
    # The url of the request
    # A JSON object to send to the specified url
    # data = {'some': 'data'}
    r = requests.post(url, json=data)  # 发送一些编码为表单形式的数据
    print(data)
    print('----')
    print(r.text)
    assert r.status_code == 200  # a 200 OK status means that your request was successful

    '''
    python3 -m register_worker --controller http://localhost:21001 --worker-name https://
    '''