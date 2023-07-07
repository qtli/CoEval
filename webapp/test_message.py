import pdb
import argparse
import json
import requests

from conversation import default_conversation


def main(args):
    if args.worker_address:
        worker_addr = args.worker_address
    else:  # None
        controller_addr = args.controller_address  # http://localhost:21001
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(controller_addr + "/get_worker_address", json={"model": args.model_name})
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    # Conversation class,
    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], args.message)
    # prompt = conv.get_prompt()
    '''
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
    ###Human: What are the key differences between renewable and non-renewable energy sources?
    ###Assistant: Renewable energy sources are those that can be replenished naturally in a relatively short amount of time, such as solar, wind, hydro, geothermal, and biomass. Non-renewable energy sources, on the other hand, are finite and will eventually be depleted, such as coal, oil, and natural gas. Here are some key differences between renewable and non-renewable energy sources:\n1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable energy sources are finite and will eventually run out.\n2. Environmental impact: Renewable energy sources have a much lower environmental impact than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, and other negative effects.\n3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically have lower operational costs than non-renewable sources.\n4. Reliability: Renewable energy sources are often more reliable and can be used in more remote locations than non-renewable sources.\n5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different situations and needs, while non-renewable sources are more rigid and inflexible.\n6. Sustainability: Renewable energy sources are more sustainable over the long term, while non-renewable sources are not, and their depletion can lead to economic and social instability.\n
    ###Human: Tell me a story with more than 1000 words.
    ###"
    '''
    prompt = 'Tell me a story with more than 1000 words.'

    headers = {"User-Agent": "fastchat Client"}
    pload = {
        "model": args.model_name,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "stop": conv.sep,  # ###
    }
    response = requests.post(args.controller_address + "/worker_generate_stream", headers=headers, json=pload, stream=True)
    # response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True)
    # pdb.set_trace()

    # print(prompt.replace(conv.sep, "\n"), end="")
    print('prompt: ', prompt)
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        print('chunk: ', chunk)
        if chunk:  # b'{"text": "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**", "error_code": 2}'
            data = json.loads(chunk.decode("utf-8"))
            # output = data["text"].split(conv.sep)[-1]
            output = data["text"]
            print(output, end="\r")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--message", type=str, default="Tell me a story with more than 1000 words.")
    args = parser.parse_args()

    main(args)

    '''
    python3 -m test_message --model gpt-3.5-turbo-0301 --controller http://localhost:21001
    '''

