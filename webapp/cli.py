import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel  # LlamaTokenizer

import datetime
import json
import os
import argparse
import pandas as pd
import openai
import openai.error as ERR

from conversation import conv_templates, SeparatorStyle
from compression import compress_module
from monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations


key_list = [
    "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",
    "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",
    "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",
    "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",
    "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",
]


default_prompt_settings = {
    'setting': "You are a helpful evaluation expert in NLP.",
     'prompt_task': "The task is to provide an answer to a Reddit forum question that is comprehensible by five-year-olds.",
     'prompt_capability': "What capabilities of the answer should there be? List directly.",
     'capability': "The answer should use simple and clear language that is easy for a five-year-old to understand.\n\n",
     'prompt_evaluation_task': "You will be given an answer for a Reddit forum question which is expected to be comprehensible by five year olds. Your task is to evaluate the summary based on a premise.\n\n",
     'evaluation_steps': "Evaluation Steps: \n1. Evaluate whether this answer satisfy the premise. Give a conclusion.\n2. Assign a score for this answer on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the premise.\n3. List evidence by quoting sentences of the answer to support your conclusion.\n",'received_evaluation': "",
}

# load checkpoints
def load_model(model_name, device, num_gpus, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


def load_openai_model(params, args):
    key_idx = 0
    openai.api_key = key_list[0]
    setting = default_prompt_settings['setting']
    if params:
        final_prompt = params["prompt"]
    else:
        prompt_task = default_prompt_settings['prompt_task']
        prompt_capability = default_prompt_settings['prompt_capability']
        final_prompt = ' '.join([prompt_task, prompt_capability])
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": setting},
                    {"role": "user", "content": final_prompt},
                ],
                max_tokens=args.length,
                temperature=args.temperature,  # 0
                top_p=args.top_p,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty
            )
            print('return response: ', response)
            result = response['choices'][0]['message']['content']
        except ERR.RateLimitError as e:
            time.sleep(5)
            key_idx += 1
            if key_idx == len(key_list):
                key_idx = 0
            openai.api_key = key_list[key_idx]
        else:
            break
    print('openai response:', result)
    yield result


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values



def main(args):
    model_name = args.model_name

    # Model
    model, tokenizer = load_model(args.model_name, args.device,
        args.num_gpus, args.load_8bit, args.debug)

    # Chat
    conv = conv_templates[args.conv_template].copy()
    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        generate_stream_func = generate_stream
        prompt = conv.get_prompt()
        skip_echo_len = len(prompt) + 1

        params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        print(f"{conv.roles[1]}: ", end="", flush=True)
        pre = 0
        for outputs in generate_stream_func(model, tokenizer, params, args.device):
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1
        print(" ".join(outputs[pre:]), flush=True)

        conv.messages[-1][-1] = " ".join(outputs)

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--load-8bit", action="store_true",
        help="Use 8-bit quantization.")
    parser.add_argument("--conv-template", type=str, default="v1",
        help="Conversation prompt template.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
