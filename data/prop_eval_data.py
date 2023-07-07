import json
import pdb
import pandas as pd
import csv
import argparse
import random
import ast
from datasets import list_datasets, load_dataset, load_from_disk
import time
import openai
import openai.error as ERR

key_list = [
    "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",
    "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",
    "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",
    "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",
    "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",
]


def prop_eli5_roc_raw():
    eli5_set = load_from_disk('eli5_hf')
    # asks = eli5_set['test_asks']
    # askh = eli5_set['test_askh']
    eli5 = eli5_set['test_eli5']
    eli5_data = {
        'title': [],
        'answers': []  # list of list
    }
    for item in eli5:
        eli5_data['title'].append(item['title'])
        eli5_data['answers'].append(item['answers']['text'])
        pdb.set_trace()
    eli5_df = pd.DataFrame(eli5_data, columns=['title', 'answers'])

    new_eli5 = []
    for idx, item in eli5_df.iterrows():
        new_eli5.append(
            {'input': "Answer the Reddit forum question in a way that is comprehensible by five-year-olds: " + item['title'],
             'output': ast.literal_eval(item['answers'])
             }
        )
    json.dump(new_eli5, open('ELI5/eli5.json', 'w'), indent=4)

    new_roc = []
    roc_hf = load_dataset("adamlin/roc_story", split='test')
    for sample in roc_hf:
        new_roc.append(
            {'input': 'Write a five-sentence story about an everyday topic \"{}\":'.format(sample['storytitle']),
            'output': sample['story'].replace('<|endoftext|> ', '')
             }
            )
        new_roc.append(
            {'input': 'Write a four-sentence story that captures daily events, continuing the given beginning: {}'.format(sample['sentence1']),
            'output': ' '.join([sample['sentence2'], sample['sentence3'], sample['sentence4'], sample['sentence5']])
             }
            )
    json.dump(new_roc, open('rocstories/rocstories.json', 'w'), indent=4)


def sample(datasets=None):
    if 'eli5' in datasets:
        eli5 = json.load(open('ELI5/eli5.json'))
        eli5_random_list = random.sample(range(1, len(eli5)), 50)  # 第二个参数是生成的不重复随机数的个数
        eli5_sample = []
        for idx in eli5_random_list:
            eli5_sample.append(eli5[idx])
        json.dump(eli5_sample, open('eli5_sample.json', 'w'), indent=4)

    if 'roc' in datasets:
        roc = json.load(open('rocstories/rocstories.json'))
        roc_random_list = random.sample(range(1, len(roc)), 50)
        roc_sample = []
        for idx in roc_random_list:
            roc_sample.append(roc[idx])
        json.dump(roc_sample, open('roc_sample.json', 'w'), indent=4)

    if 'self_instruct' in datasets:
        with open('self_instruct/user_oriented_instructions.jsonl') as f:
            self_instruct = []
            for line in f.readlines():
                data = json.loads(line)
                if len(data["instances"])>=1:
                    self_instruct.append(data)
        instruct_random_list = random.sample(range(1, len(self_instruct)), 50)  # 第二个参数是生成的不重复随机数的个数
        # instruct_random_list = range(len(self_instruct))  # 第二个参数是生成的不重复随机数的个数

        instruct_sample = []
        for idx in instruct_random_list:
            instruct_sample.append(self_instruct[idx])
        json.dump(instruct_sample, open('self_instruct_all.json', 'w'), indent=4)


def simplify_text(text, dataset):
    if 'eli5' in dataset:
        text = text.lstrip("Answer the Reddit forum question in a way that is comprehensible by five-year-olds: ")
    if 'self_instruct' in dataset:
        text = text.rstrip("\nOutput:")
    return text

def merge(dict1, dict2):
    for k in dict1:
        if k == 'criteria':
            continue
        dict2[k].extend(dict1[k])
    return dict2

def model_prediction(input_file, start_idx, end_idx, max_num_tokens, model_names, output_files):
    key_idx = 0
    openai.api_key = key_list[0]
    temperature = 0.7
    top_p = 1.0
    frequency_penalty = 0.0
    presence_penalty = 1

    prompted_texts = []
    for item in json.load(open(input_file))[start_idx: end_idx]:
        prompted_texts.append(item['input'])
    max_tokens = max_num_tokens

    for model_idx, model in enumerate(model_names):
        print(f'input_file: {input_file}, model: {model}')
        f = open(output_files[model_idx], 'a')
        for cur_idx, text in enumerate(prompted_texts):
            if 'turbo' not in model:
                response = openai.Completion.create(
                    model=model,
                    prompt=text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                print(response)
                f.write(json.dumps({"input": text, "prediction": response["choices"][0]["text"].strip('\n')}) + '\n')
            else:
                while cur_idx <= len(prompted_texts):
                    try:
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": ""},
                                {"role": "user", "content": text},
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature,  # 0
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty
                        )
                        print(response['choices'][0]['message']['content'])
                        f.write(json.dumps({"input": text, "prediction": response['choices'][0]['message']['content'].strip('\n')}) + '\n')
                    except ERR.RateLimitError as e:
                        time.sleep(5)
                        key_idx += 1
                        if key_idx == len(key_list):
                            key_idx = 0
                        openai.api_key = key_list[key_idx]
                    else:
                        print('continue the invoke')
                        break

def combine_prediction_with_criteria(input_files, datasets, model_files, start_idx=0, end_idx=0, to_file=""):
    combine = {}
    for idx, dataset in enumerate(datasets):
        input_file = json.load(open(input_files[idx]))

        curie_f = open(model_files[dataset]['curie'])
        turbo_f = open(model_files[dataset]['turbo'])

        if '.json' in model_files['davinci_002']:
            davinci_002_f = json.load(open(model_files[dataset]['davinci_002']))
            davinci_002 = [item["prediction"] for item in davinci_002_f]
            davinci_002_ipt = [item["input"] for item in davinci_002_f]
        else:
            davinci_002_f = open(model_files[dataset]['davinci_002'])
            davinci_002 = [json.loads(line)["prediction"] for line in davinci_002_f.readlines()]
            davinci_002_ipt = [json.loads(line)["input"] for line in davinci_002_f.readlines()]

        curie = [json.loads(line)["prediction"] for line in curie_f.readlines()]
        turbo = [json.loads(line)["prediction"] for line in turbo_f.readlines()]

        curie_f = open(model_files[dataset]['curie'])
        turbo_f = open(model_files[dataset]['turbo'])
        curie_ipt = [json.loads(line)["input"] for line in curie_f.readlines()]
        turbo_ipt = [json.loads(line)["input"] for line in turbo_f.readlines()]
        assert davinci_002_ipt[start_idx:end_idx] == curie_ipt[start_idx:end_idx] == turbo_ipt[start_idx:end_idx]

        all_source = []
        all_target = []
        all_model_name = []

        for specify_idx in range(len(input_file))[start_idx:end_idx]:
            item = input_file[specify_idx]
            source_input = simplify_text(item['input'], dataset)
            if isinstance(item['output'], list):
                human = item['output'][0]
            else:
                human = item['output']

            item_outputs = [human, davinci_002[specify_idx], curie[specify_idx], turbo[specify_idx]]
            model_names = ["human", "davinci_002", "curie_001", "turbo"]
            all_source.extend([source_input]*len(model_names))

            c = list(zip(item_outputs, model_names))
            random.shuffle(c)
            item_outputs, model_names = zip(*c)

            all_target.extend(item_outputs)
            all_model_name.extend(model_names)

        if 'eli5' in dataset:
            combine[dataset] = {
                "task_info": ["The task is to provide an answer to a Reddit forum question that is comprehensible by five-year-olds."] * len(all_source),
                "source_name": ["reddit question"] * len(all_source),
                "target_name": ["answer"] * len(all_source),
                "source": all_source,
                "target": all_target,
                "model_name": all_model_name,
                "criteria": [
                    "1. Use simple and easy-to-understand language.",
                    "2. Use examples or analogies that are relatable to a five-year-old’s experience.",
                    "3. Answers should be factually correct and cannot have subtly incorrect or fabricated information.",
                    "4. Be easy to follow and logically coherent.",
                    "5. All things considered, answers should be helpful to the person who asked this question."
                ]
            }
        if dataset == "roc":
            combine['roc'] = {
                "task_info": ["The task is to a story that captures daily events, starting with a given beginning."] * len(all_source),
                "source_name": ["beginning"] * len(all_source),
                "target_name": ["story"] * len(all_source),
                "source": all_source,
                "target": all_target,
                "model_name": all_model_name,
                "criteria": [
                    "1. Clarity: The story should be clear and easy to understand, with no confusing or ambiguous elements.",
                    "2. Relevance: The story should be relevant to the given beginning and the daily events it aims to capture",
                    "3. Coherence: The story should have a logical flow and provide closure.",
                    "4. Length: The story should be an appropriate length for the given task.",
                    "5. Engagement: The story should be engaging from beginning to end.",
                    "6. Character development: The characters in the story should be well-developed.",
                    "7. Grammaticality: The story should be grammatically correct"
                ]
            }
        if dataset == "self_instruct":
            combine['self_instruct'] = {
                "task_info": ["The task is to generate a response to a user's instruction."] * len(all_source),
                "source_name": ["instruction"] * len(all_source),
                "target_name": ["response"] * len(all_source),
                "source": all_source,
                "target": all_target,
                "model_name": all_model_name,
                "criteria": [
                    "1. Relevance: The response should be relevant to the user's instruction or query.",
                    "2. Clarity: The response should be clear and easy to understand.",
                    "3. Correctness: The response should be factually correct and free of errors.",
                    "4. Politeness: The response should be polite and respectful.",
                    "5. Completeness: The response should provide a comprehensive and detailed answer to the user's instruction.",
                ]
            }
        print(f'{dataset}: {len(all_source)}')
        # full = merge(combine[dataset], full)
    json.dump(combine, open(f'{to_file}.json', 'w'), indent=4)
    # df = pd.DataFrame(data=full, columns=['task_info', 'source_name', 'target_name', 'model_name', 'source', 'target'])
    # df.to_csv(f'/Users/liqintong/Desktop/mypapers/benchmarking_llm_textgen/inference_llms/webapp/test/{to_file}.csv')


def firstletterupper(text):
    return text[0].upper() + text[1:]

def combine_eval_query(task_info, src_name, src, tgt_name, tgt, eval_cap):
    if tgt_name.lower()[0] in ['a', 'e', 'i', 'o', 'u']:
        article = "an"
    else:
        article = "a"
    task_info = " ".join([task_info, "You need to evaluate {} {} candidate based on a premise.".format(article, tgt_name.lower())]) + "\n\n"
    premise = "Premise: {}\n\n".format(firstletterupper(eval_cap))
    src = "{}: {}\n\n".format(firstletterupper(src_name), firstletterupper(src))
    tgt = "{}: {}\n\n".format(firstletterupper(tgt_name), firstletterupper(tgt))
    steps = "Evaluation Steps: \n1. Evaluate whether this {} satisfy the premise. Give a conclusion.\n2. Assign a score for this answer on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the premise.\n3. List evidence by quoting sentences of the {} to support your conclusion.".format(tgt_name.lower(), tgt_name.lower())
    final = task_info + premise + src + tgt + steps
    print('combine_eval_query: ')
    print(final)
    return final

def generate_eval_result(input_file, datasets, max_num_tokens, temperature, top_p, frequency_penalty, presence_penalty,
                         model_name, sample_start, criteria_start, sample_end):
    key_idx = 0
    openai.api_key = key_list[0]

    for dataset in datasets:
        prompted_texts = json.load(open(input_file))[dataset]
        criteria = prompted_texts['criteria']
        prompted_texts_size = len(prompted_texts['task_info'])
        max_tokens = max_num_tokens
        model = model_name

        sample_start, criteria_start, sample_end = sample_start, criteria_start, sample_end
        last_eval_start = criteria_start
        eval_idx = 0

        f = open(f"{dataset}_eval_result.txt", 'a')
        for idx in range(prompted_texts_size)[sample_start:sample_end]:
            if 'turbo' not in model:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompted_texts,  # todo
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                # print(response)
            else:
                while idx <= (range(prompted_texts_size)[sample_start:sample_end])[-1] and last_eval_start < len(criteria):
                    try:
                        task_info = prompted_texts['task_info'][idx]
                        src_name = prompted_texts['source_name'][idx]
                        src = prompted_texts['source'][idx]
                        model_name = prompted_texts['model_name'][idx]
                        tgt_name = prompted_texts['target_name'][idx]
                        tgt = prompted_texts['target'][idx]

                        for eval_idx in range(len(criteria))[last_eval_start:]:
                            eval_cap = criteria[eval_idx]
                            combine_prompt = combine_eval_query(task_info, src_name, src, tgt_name, tgt, eval_cap)
                            response = openai.ChatCompletion.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "You are a helpful evaluation expert in NLP."},
                                    {"role": "user", "content": combine_prompt},
                                ],
                                max_tokens=max_num_tokens,
                                temperature=temperature,  # 0
                                top_p=top_p,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty
                            )
                            print(response['choices'][0]['message']['content'])
                            f.write(json.dumps({"idx": idx,
                                                "model_name": model_name,
                                                "source": src,
                                                'eval_cap': eval_cap,
                                                "eval_result": response['choices'][0]['message']['content'].strip('\n')}) + '\n')
                    except ERR.RateLimitError as e:
                        last_eval_start = eval_idx
                        time.sleep(5)
                        key_idx += 1
                        if key_idx == len(key_list):
                            key_idx = 0
                        openai.api_key = key_list[key_idx]
                    else:
                        last_eval_start = 0
                        print('============continue==============')
                        break


def combine_criteria_eval(file):
    input_data = json.load(open(file))

    for dataset in input_data:
        c_eval = open(f"{dataset}_eval_result.txt")
        idx_to_models = {}

        cri = input_data[dataset]['criteria']
        input_data[dataset]['criteria_kv'] = {}
        for c in cri:
            input_data[dataset]['criteria_kv'][c] = []

        for model_idx, m in enumerate(input_data[dataset]["model_name"]):
            idx_to_models[model_idx] = m

        for line_idx, line in enumerate(c_eval.readlines()):
            line_data = json.loads(line)
            this_c = line_data['eval_cap']
            this_eval_result = line_data['eval_result']
            input_data[dataset]['criteria_kv'][this_c].append(this_eval_result)

    json.dump(input_data, open(file, 'w'), indent=4)

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="prepare_test_instance")
    parser.add_argument("--model_name", default=None, type=str, required=False, help="openai model name")
    parser.add_argument("--input_file", default=None, type=str, required=False, help="input file")
    parser.add_argument("--input_files", default=None,  type=str, required=False, help="multiple files to be considered for generating")
    parser.add_argument("--sample_number", default=None, type=int, required=False, help="sample number from the input file")
    parser.add_argument("--output_file", default=None, type=str, required=False, help="save your result")
    parser.add_argument("--output_files", default=None, type=str, required=False, help="save your results to multiple files.")
    parser.add_argument("--prompt", type=str, default="")

    parser.add_argument("--max_num_tokens", type=int, default=512)
    parser.add_argument("--stop_token", type=str, default='</s>', help="Token at which text generation is stopped")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 1.0 has no effect, lower tend toward greedy sampling",)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--generate_num", default=50, type=int, help="generation number")
    parser.add_argument("--start_idx", default=0, type=int, help="generation number")
    parser.add_argument("--end_idx", default=100, type=int, help="generation number")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = add_args()

    if args.stage == "prepare_test_instance":
        prop_eli5_roc_raw()
        sample(datasets=['eli5', 'roc', 'self_instruct'])


    if args.stage == "model_prediction":
        input_files = ["ELI5/eli5_sample.json", "rocstories/roc_sample.json", "self_instruct/self_instruct_sample.json"]
        output_files = [['ELI5/eli5_turbo_50.txt', 'ELI5/eli5_davinci_002_50.json', 'ELI5/eli5_curie_50.txt'],
                        ['rocstories/roc_turbo_50.txt', 'rocstories/roc_davinci_002_50.txt', 'rocstories/roc_curie_50.txt'],
                        ['self_instruct/self_instruct_turbo_50.txt', 'self_instruct/self_instruct_davinci_002_50.txt', 'self_instruct/self_instruct_curie_50.txt']]
        model_names = ["gpt-3.5-turbo-0301", "text-davinci-002", "text-curie-001"]
        for idx, ipt_file in enumerate(input_files):
            model_prediction(input_file=ipt_file,
                             output_files=output_files[idx],
                             start_idx=0,
                             end_idx=50,
                             model_names=model_names,
                             max_num_tokens=args.max_num_tokens)

    if args.stage == "combine_model_prediction":
        combine_prediction_with_criteria(  # shuffle model predictions
            input_files=["ELI5/eli5_sample.json", "rocstories/roc_sample.json", "self_instruct/self_instruct_sample.json"],
            datasets=['eli5', 'roc', 'self_instruct'],
            model_files={
                "eli5": {
                    "davinci_002": "ELI5/eli5_davinci_002_50.json",
                    "curie": "ELI5/eli5_curie_50.txt",
                    "turbo": 'ELI5/eli5_turbo_50.txt'},
                "roc": {
                    "davinci_002": "rocstories/roc_davinci_002_50.txt",
                    "curie": "rocstories/roc_curie_50.txt",
                    "turbo": 'rocstories/roc_turbo_50.txt'
                },
                "self-instruct": {
                    "davinci_002": "self_instruct/self_instruct_davinci_002_50.txt",
                    "curie": "self_instruct/self_instruct_curie_50.txt",
                    "turbo": 'self_instruct/self_instruct_turbo_50.txt'
                }
            },
            start_idx=0,
            end_idx=50,
            to_file="combine_data.json")


    if args.stage == "llm_eval":
        generate_eval_result(input_file="combine_data.json", datasets=['eli5', 'roc', 'self_instruct'],
                             max_num_tokens=args.max_num_tokens, temperature=args.temperature, top_p=args.top_p, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty,
                             model_name=args.model_name, sample_start=0, sample_end=200, criteria_start=0)


        combine_criteria_eval(file="combine_data.json")