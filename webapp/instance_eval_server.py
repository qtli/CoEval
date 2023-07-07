import copy
import os
import argparse
import pdb
from collections import defaultdict
import datetime
import json
import random
import string
import numpy as np
import time
import uuid
import pandas as pd
import gradio as gr
import requests

from evaluator import default_collaboration, collab_templates, SeparatorStyle
from utils import LOGDIR, build_logger, server_error_msg, violates_moderation, moderation_msg
from gradio_content import priority, css, notice_markdown, criteria_query_title, criteria_evaluation_title_t2, \
    learn_more_markdown, sep_line, get_window_url_params, cap_examples, eval_examples

logger = build_logger("eval_instance", "eval_instance.log")
headers = {"User-Agent": "fastchat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)
worker_id_to_progress = {}

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models

def read_back_file():
    eval_data_dict = pro_eval_data(args.eval_data_name)
    eval_data_all = gr.State(value=eval_data_dict)
    task_to_eval_data = gr.State(value=json.load(open(args.eval_data_name)))
    all_tasks = list(eval_data_dict.keys())
    return eval_data_all, task_to_eval_data, all_tasks

def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=False)

    task_pointer = 0
    example_pointer = 0
    total_example_pointer = 0
    eval_start = 0
    eval_df_all = pd.DataFrame(
        data={'Instance ID': [], 'Task information': [], 'Source': [], 'Target': [], 'Criteria': [],
              'Final Score': [], 'Evaluation': [],
              'APPROVE': [], 'DELETE': [], 'REVISE or ADD': [], 'New Evaluation': []},
        columns=['Instance ID', 'Task information', 'Source', 'Target', 'Criteria', 'Final Score', 'Evaluation',
                 'APPROVE', 'DELETE',
                 'REVISE or ADD', 'New Evaluation'])

    eval_data_all, task_to_eval_data, all_tasks = read_back_file()

    state = default_collaboration.copy()
    return (state,
            dropdown_update,
            # gr.Textbox.update(visible=True),
            gr.DataFrame.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Row.update(visible=True),
            task_pointer, example_pointer, total_example_pointer, eval_start, eval_df_all,
            eval_data_all, task_to_eval_data, all_tasks)

def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()

    task_pointer = 0
    example_pointer = 0
    total_example_pointer = 0
    eval_start = 0
    eval_df_all = pd.DataFrame(
        data={'Instance ID': [], 'Task information': [], 'Source': [], 'Target': [], 'Criteria': [],
              'Final Score': [], 'Evaluation': [],
              'APPROVE': [], 'DELETE': [], 'REVISE or ADD': [], 'New Evaluation': []},
        columns=['Instance ID', 'Task information', 'Source', 'Target', 'Criteria', 'Final Score', 'Evaluation',
                 'APPROVE', 'DELETE',
                 'REVISE or ADD', 'New Evaluation'])

    eval_data_all, task_to_eval_data, all_tasks = read_back_file()

    state = default_collaboration.copy()
    dropdown_update = gr.Dropdown.update(visible=False)
    return (state,
            dropdown_update,
            gr.DataFrame.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Row.update(visible=True),
            task_pointer, example_pointer, total_example_pointer, eval_start, eval_df_all,
            eval_data_all, task_to_eval_data, all_tasks)

def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_collaboration.copy()
    return (state,
            pd.DataFrame(
                {"Criteria": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Criteria": [""]},
                columns=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"]),
            "",
            "",
            "") + (enable_btn,) * 3

def clear_eval_history(request: gr.Request):
    # state, eval_output, eval_src_textbox, eval_tgt_textbox, final_score
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_collaboration.copy()
    return (state,
            pd.DataFrame({"Evaluation": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""],
                          "New Evaluation": [""]},
                         columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"]),
            gr.Number.update(value=None),
            gr.Radio.update(choices=["criteron"])) + (enable_btn,) * 3

def post_process_code(code, cap_or_eval='cap'):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)

    while ('\n\n' in code):
        code = code.replace('\n\n', '\n')
    code_list = code.split('\n')
    if cap_or_eval == 'cap':
        ids = [*range(1, len(code_list) + 1)]
        code_df = pd.DataFrame({"Criteria": code_list,
                                "APPROVE ✅": ["" for _ in range(len(code_list))],
                                "DELETE ❎": ["" for _ in range(len(code_list))],
                                "REVISE 🔀 or ADD 🆕": ["" for _ in range(len(code_list))],
                                "New Criteria": ["" for _ in range(len(code_list))],
                                },
                               columns=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"])
    else:
        code_df = pd.DataFrame({"Evaluation": code_list,
                                "APPROVE ✅": ["" for _ in range(len(code_list))],
                                "DELETE ❎": ["" for _ in range(len(code_list))],
                                "REVISE 🔀 or ADD 🆕": ["" for _ in range(len(code_list))],
                                "New Evaluation": ["" for _ in range(len(code_list))],
                                },
                               columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"])
    return code, code_df

def change_textbox(change_text):
    return gr.update(visible=True, label=change_text)

def http_bot_eval(state, model_name, temperature, max_new_tokens):
    '''
    then(http_bot, inputs=[state, model_selector, temperature, max_output_tokens],
                   outputs=[state, eval_output] + btn_list)
    '''
    template_name = "v2"
    new_state = collab_templates[template_name].copy()
    new_state.conv_id = uuid.uuid4().hex
    new_state.append_message(new_state.roles[0], state.messages[-2][1])
    new_state.append_message(new_state.roles[1], None)
    state = new_state

    empty_df = pd.DataFrame(
        {"Evaluation": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]},
        columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"])

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    prompt = state.get_prompt_eval()
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
    }
    logger.info(f"==== request ====\n{pload}")
    state.messages[-1][-1] = "▌"  # cursor

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True, timeout=100)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:  # success
                    output = data["text"].strip()
                    output, output_df = post_process_code(output, cap_or_eval='eval')
                    state.messages[-1][-1] = output + "▌"
                    yield state, output, output_df
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield state, output, empty_df
                    return
                time.sleep(0.02)
    except requests.exceptions.RequestException as e:
        # usually vpn -> connection issue
        state.messages[-1][-1] = server_error_msg + f" (error_code: 4)"
        yield state, e, empty_df
        return
    state.messages[-1][-1] = state.messages[-1][-1][:-1] # remove cursor

def new_eval_criteron(criteria, task_info, src_name, tgt_name, src, tgt, state, example_pointer, task_pointer, task_to_eval_data, choose_task_box, invoke_mode):
    cur_task_data = task_to_eval_data.value[choose_task_box]

    if invoke_mode == 'offline':
        eval_output = cur_task_data['criteria_kv'][criteria][example_pointer-1].strip()
        eval_output, eval_output_df = post_process_code(eval_output, cap_or_eval='eval')
    else:  # online
        new_text = state.combine_eval_query(task_info, src_name, src, tgt_name, tgt, criteria)
        template_name = "v2"
        new_state = collab_templates[template_name].copy()
        new_state.conv_id = uuid.uuid4().hex
        new_state.append_message(new_state.roles[0], new_text)
        new_state.append_message(new_state.roles[1], None)
        state = new_state
        result = [item for item in http_bot_eval(state, model_name=args.model_name, temperature=0, max_new_tokens=256)]
        state, eval_output, eval_output_df = result[0]

    lt = time.localtime(time.time())
    to_time = time.strftime("%Y-%m-%d %H:%M:%S", lt)
    eval_start = to_time
    return state, eval_output_df, eval_start

def record_eval_correction(eval_result, task_info, src_name, tgt_name, eval_src, eval_tgt, eval_cap, final_score,
                           workerid, instance_id, task_pointer, example_pointer, total_example_pointer,
                           eval_start, eval_df_all, choose_task_box):
    global worker_id_to_progress
    lt = time.localtime(time.time())
    to_time = time.strftime("%Y-%m-%d %H:%M:%S", lt)
    eval_end = to_time

    ret = eval_result.to_json(orient="values")
    parsed = json.loads(ret)
    parsed.insert(0, {'worker_id': workerid, 'task_info': task_info, 'src_name': src_name, 'tgt_name': tgt_name,
                      'eval_src': eval_src, 'eval_tgt': eval_tgt, 'eval_cap': eval_cap, 'final_score': final_score,
                      'task_pointer': task_pointer, 'task_example_pointer': example_pointer,
                      'example_pointer': total_example_pointer, 'eval_start_time': eval_start,
                      'eval_end_time': eval_end})
    ret = json.dumps(parsed)

    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-{workerid}-evaluation-correction-{choose_task_box}.json")
    with open(name, "a") as fout:
        fout.write(ret + "\n")

    empty_df = pd.DataFrame(
        {"Evaluation": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]},
        columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"])


    this_task_info = [task_info] + [""] * (len(eval_result) - 1)
    this_source = [eval_src] + [""] * (len(eval_result) - 1)
    this_target = [eval_tgt] + [""] * (len(eval_result) - 1)
    this_criteron = [eval_cap] + [""] * (len(eval_result) - 1)
    this_final_score = [final_score] + [""] * (len(eval_result) - 1)
    this_instance_id = [instance_id] + [""] * (len(eval_result) - 1)

    for idx in range(len(eval_result)):
        # 增加新的一列
        new_row = [this_instance_id[idx], this_task_info[idx],
                   this_source[idx], this_target[idx],
                   this_criteron[idx], this_final_score[idx],
                   eval_result['Evaluation'][idx],
                   eval_result['APPROVE ✅'][idx], eval_result['DELETE ❎'][idx],
                   eval_result['REVISE 🔀 or ADD 🆕'][idx],
                   eval_result['New Evaluation'][idx]]
        eval_df_all.loc[len(eval_df_all.index)] = new_row

    if workerid not in worker_id_to_progress:
        worker_id_to_progress[workerid] = [{"instance_id": instance_id, "time": eval_end}]
    else:
        worker_id_to_progress[workerid].append({"instance_id": instance_id, "time": eval_end})
    return empty_df, gr.Number.update(value=None), eval_df_all

def next_example(state, instance_id_textbox, example_pointer, total_example_pointer, eval_data_all, task_to_eval_data, choose_task_box):
    '''
    eval_next_btn.click(fn=next_example,
                    inputs=[state, instance_id_textbox, example_pointer, total_example_pointer,
                            eval_data_all, task_to_eval_data, choose_task_box],
                    outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn,
                             textbox, src_name_textbox, tgt_name_textbox,
                             eval_criteria, eval_last_btn, annotation_finished, annotation_workload,
                             instance_id_textbox, example_pointer, total_example_pointer])
    '''
    cur_task_data = task_to_eval_data.value[choose_task_box]
    if example_pointer == len(cur_task_data['source']):
        source = ""
        target = ""
        task_info, source_name, target_name = "", "", ""
        btn = disable_btn  # disable next button
        cap_choices = [None]
    else:
        source = cur_task_data['source'][example_pointer]
        target = cur_task_data['target'][example_pointer]
        source_name = cur_task_data['source_name'][example_pointer]
        target_name = cur_task_data['target_name'][example_pointer]
        task_info = cur_task_data['task_info'][example_pointer]

        cap_choices = cur_task_data['criteria']
        btn = enable_btn

        example_pointer += 1
        total_example_pointer += 1

    if example_pointer == 1:
        last_btn = disable_btn
    else:
        last_btn = enable_btn

    instance_id_textbox += 1
    annotation_finished = (example_pointer-1)
    annotation_workload = (eval_data_all.value[choose_task_box]-annotation_finished)

    return state, source, target, btn, \
           task_info, source_name, target_name, \
           gr.Radio.update(value=cap_choices[0], choices=cap_choices), last_btn, \
           annotation_finished, annotation_workload, \
           instance_id_textbox, example_pointer, total_example_pointer

def last_example(state, instance_id_textbox, example_pointer, total_example_pointer, eval_data_all, task_to_eval_data, choose_task_box):
    example_pointer -= 2
    total_example_pointer -= 2

    cur_task_data = task_to_eval_data.value[choose_task_box]
    source = cur_task_data['source'][example_pointer]
    target = cur_task_data['target'][example_pointer]
    source_name = cur_task_data['source_name'][example_pointer]
    target_name = cur_task_data['target_name'][example_pointer]
    task_info = cur_task_data['task_info'][example_pointer]
    cap_choices = cur_task_data['criteria']
    btn = enable_btn

    example_pointer += 1  # nex instance
    total_example_pointer += 1

    if example_pointer == 1:
        last_btn = disable_btn
    else:
        last_btn = enable_btn

    instance_id_textbox -= 1

    empty_df = pd.DataFrame(
        {"Evaluation": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]},
        columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"])
    return state, source, target, btn, (eval_data_all.value[choose_task_box] - total_example_pointer), total_example_pointer, \
           task_info, source_name, target_name, \
           gr.Radio.update(value=cap_choices[0], choices=cap_choices), last_btn, instance_id_textbox, \
           example_pointer, total_example_pointer, empty_df

def jump_example(state, instance_id, eval_data_all, task_to_eval_data, choose_task_box):
    '''
            jump_btn.click(fn=jump_example,
                        inputs=[state, jump_instance_id_textbox, task_pointer],
                        outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn,
                        annotation_workload, annotation_finished, textbox, src_name_textbox, tgt_name_textbox,
                        eval_criteria, eval_last_btn, instance_id_textbox,
                        example_pointer, total_example_pointer])

    '''
    example_pointer = (int(instance_id)-1)  # todo index and no.
    total_example_pointer = (int(instance_id)-1)

    cur_task_data = task_to_eval_data.value[choose_task_box]
    source = cur_task_data['source'][example_pointer]
    target = cur_task_data['target'][example_pointer]
    source_name = cur_task_data['source_name'][example_pointer]
    target_name = cur_task_data['target_name'][example_pointer]
    task_info = cur_task_data['task_info'][example_pointer]
    cap_choices = cur_task_data['criteria']
    btn = enable_btn

    example_pointer += 1  # next instance
    total_example_pointer += 1

    if example_pointer == 1:
        last_btn = disable_btn
    else:
        last_btn = enable_btn

    cur_instance_id = instance_id
    annotation_workload = (eval_data_all.value[choose_task_box] - total_example_pointer)

    empty_df = pd.DataFrame(
        {"Evaluation": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]},
        columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"])

    return state, source, target, btn, annotation_workload, total_example_pointer, \
           task_info, source_name, target_name, \
           gr.Radio.update(value=cap_choices[0], choices=cap_choices), last_btn, cur_instance_id, \
           example_pointer, total_example_pointer, empty_df

def change(cnt):
    data = [[1, 2], [3, 4]]
    return data

def clear():
    data = pd.DataFrame(data={"A": [""], "B": [""]}, columns=["A", "B"])
    return (data, data)

def export_eval(worker_id, eval_df_all):
    t = datetime.datetime.now()
    random_string = ''.join(random.choices(string.ascii_letters, k=6))
    if os.path.exists('annotation_data/annotator_export/') is False:
        os.makedirs('annotation_data/annotator_export/')

    eval_df_all.to_excel(f"annotation_data/annotator_export/{t.year}-{t.month:02d}-{t.day:02d}-{worker_id}-evaluation-{random_string}.xlsx",
                         index=None,
                         header=True)

    return gr.File.update(value=f"annotation_data/annotator_export/{t.year}-{t.month:02d}-{t.day:02d}-{worker_id}-evaluation-{random_string}.xlsx",
                          visible=True)

def check_annotation_data(ipt_files, annotation_data, instance_to_num_data, worker_name="", tofile=""):
    result = {}
    result[worker_name] = {}

    cur_model_idx = 0
    cur_model_criteria = 0

    for ipt_file in ipt_files:
        with open(ipt_file) as f:
            for idx, item in enumerate(f.readlines()):
                item = json.loads(item)

                if worker_name not in item[0]["worker_id"]:
                    print('ERR')
                    continue

                item_format = {
                    item[0]['src_name']: item[0]['eval_src'],
                    item[0]['tgt_name']: item[0]['eval_tgt'],
                    'criteria': {}
                }
                combine_instance = f"[source]: {item[0]['eval_src']}\t[target]: {item[0]['eval_tgt']}"
                if combine_instance in instance_to_num_data:
                    size = instance_to_num_data[combine_instance]
                    if len(size) == 1:
                        item_format['model'] = annotation_data['model_name'][int(size[0])]
                        instance_id = int(size[0]) + 1  # start from 1

                    else:
                        item_format['model'] = annotation_data['model_name'][item[0]["task_example_pointer"] - 1]
                        instance_id = int(item[0]["task_example_pointer"])

                        cur_model_criteria += 1  # n criteria for each instance
                        if cur_model_criteria == len(annotation_data["criteria"]):
                            cur_model_criteria = 0
                            cur_model_idx += 1

                        if cur_model_idx == len(size):
                            cur_model_idx = 0
                else:
                    continue

                if instance_id not in result[worker_name]:
                    result[worker_name][instance_id] = item_format  # iterate all criteria

                criterion = item[0]['eval_cap']

                if criterion not in result[worker_name][instance_id]['criteria']:
                    if item[0]['final_score'] is not None:  # NON NULL
                        if int(item[0]['final_score']) != 0:  # NON 0
                            final_score = item[0]['final_score']
                        else:
                            final_score = 1.0

                    result[worker_name][instance_id]['criteria'][criterion] = {
                        'final_score': final_score,
                        'start_time': item[0]['eval_start_time'],
                        'end_time': item[0]['eval_end_time'],
                        'eval_result': item[1:]
                    }
                else:
                    # combine multiple records
                    eval_result = item[1:]
                    for eval_r_idx, eval_r in enumerate(eval_result):
                        if eval_r_idx >= len(result[worker_name][instance_id]['criteria'][criterion]['eval_result']):
                            # add operation
                            result[worker_name][instance_id]['criteria'][criterion]['eval_result'].append(eval_r)
                        else:
                            if eval_r[1:] != ["", "", "", ""]:
                                result[worker_name][instance_id]['criteria'][criterion]['eval_result'][
                                    eval_r_idx] = eval_r

    new_result = {}
    for worker_n in result:
        int_docs_info = {int(k): v for k, v in result[worker_n].items()}
        new_result[worker_n] = dict(sorted(int_docs_info.items()))
    json.dump(new_result, open(tofile, 'w'), indent=4)

def analysis_annotator_data(file, worker_name, all_criteria, data_size, to_file):
    '''
    make sure all instances and all criteria has been labeled
    '''
    if "4. Length: The story should be an appropriate length for the given task." in all_criteria:
        all_criteria.remove("4. Length: The story should be an appropriate length for the given task.")

    data = json.load(open(file))
    miss = []
    with open(to_file, 'w') as f:
        for k in data[worker_name].keys():
            if len(data[worker_name][k]["criteria"].keys()) != len(all_criteria):
                for c in all_criteria:
                    if c not in data[worker_name][k]["criteria"].keys():
                        # certain criteria not be used
                        f.write(f"[no criteria] instance_id: {int(k)},  missed criteria: {c}\n")

            for criteria in data[worker_name][k]["criteria"]:
                score = data[worker_name][k]["criteria"][criteria]["final_score"]
                if score is None:
                    f.write(f"[score NULL] instance_id: {int(k)}, criteria: {criteria}, score error: {score}\n")
                else:
                    score_int = int(score)
                    if score_int < 1 or score_int > 5:
                        f.write(f"[score range error] instance_id: {int(k)}, criteria: {criteria},  score error: {score}\n")
                eval_result = data[worker_name][k]["criteria"][criteria]["eval_result"]
                for item in eval_result:
                    # no action for an evaluation result
                    if item[1:] == ["", "", "", ""]:
                        f.write(f"[no action] instance_id: {int(k)},  criteria: {criteria}, eval result: {item}\n")
                    if (item[3] == "" or item[4] != "") is False:
                        f.write(f"[no new evaluation] instance_id: {int(k)},  criteria: {criteria}, eval result: {item}\n")
        for count_idx in range(1, data_size+1):
            if str(count_idx) not in data[worker_name].keys():
                miss.append(count_idx)
        # some instances are not evaluted
        f.write(f'[no instance] Miss instance ID: {miss}\n')
        f.write('\n\n')

def iterate_files(path, worker_id, task_name):
    file_list = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f'-{worker_id}-evaluation-correction' in f and task_name in f:
                file_list.append(os.path.join(path, f))
    logger.info('{}_file_list: {}'.format(worker_id, file_list))
    return file_list

def export_all(export_worker_id_textbox, data_type):
    '''
    worker_id_to_progress: {"worker": [ {"instance_id": "idx1", "time": "time1"}, {"idx2": "time2"} ]}
    worker_id_to_eval: {"worker": [ ["idx1": eval_result1], ["idx2": eval_result2] ]}
    '''
    global worker_id_to_progress
    new_worker_id_to_progress = {}
    for worker in worker_id_to_progress:
        progress = worker_id_to_progress[worker]
        sorted_progress = sorted(progress, key=lambda x: x['instance_id'])
        new_worker_id_to_progress[worker] = sorted_progress

    if os.path.exists('export_data/') is False:
        os.makedirs('export_data')

    json.dump(new_worker_id_to_progress, open(f'export_data/{export_worker_id_textbox}_progress.json', 'w'), indent=4)

    result1 = gr.File.update(value=f'export_data/{export_worker_id_textbox}_progress.json', visible=True)
    data_size = 200
    annotation_data = json.load(open(args.eval_data_name))[data_type]
    if data_type == 'eli5':
        instance_to_num_data = json.load(open("../data/ELI5/eli5_instance_to_num.json"))
        data_size = 150
    elif data_type == 'roc':
        instance_to_num_data = json.load(open("../data/rocstories/roc_instance_to_num.json"))
    elif data_type == "self_instruct":
        instance_to_num_data = json.load(open("../data/self_instruct/self_instruct_instance_to_num.json"))
        data_size = 200


    all_criteria = annotation_data['criteria']


    worker_files = iterate_files(path="annotation_data/", worker_id=export_worker_id_textbox, task_name=data_type)
    check_annotation_data(ipt_files=worker_files,
                          worker_name=export_worker_id_textbox,
                          annotation_data=annotation_data,
                          instance_to_num_data=instance_to_num_data,
                          tofile=f'export_data/{export_worker_id_textbox}_eval_result_progress.json')

    analysis_annotator_data(file=f'export_data/{export_worker_id_textbox}_eval_result_progress.json',
                            worker_name=export_worker_id_textbox,
                            all_criteria=all_criteria,
                            data_size=data_size,
                            to_file=f'export_data/{export_worker_id_textbox}_error_report.txt')

    result3 = gr.File.update(value=f'export_data/{export_worker_id_textbox}_error_report.txt', visible=True)

    return result1, gr.File.update(value=f'export_data/{export_worker_id_textbox}_eval_result_progress.json', visible=True), result3

def choose_cur_task(worker_id, eval_data_all, task_name):
    task_size = eval_data_all.value[task_name]

    example_pointer = 0
    total_example_pointer = 0
    eval_start = 0
    eval_df_all = pd.DataFrame(
        data={'Instance ID': [], 'Task information': [], 'Source': [], 'Target': [], 'Criteria': [],
              'Final Score': [], 'Evaluation': [],
              'APPROVE': [], 'DELETE': [], 'REVISE or ADD': [], 'New Evaluation': []},
        columns=['Instance ID', 'Task information', 'Source', 'Target', 'Criteria', 'Final Score', 'Evaluation',
                 'APPROVE', 'DELETE',
                 'REVISE or ADD', 'New Evaluation'])

    instance_id_textbox = 0
    jump_instance_id_textbox = 0
    eval_criteria = gr.Radio.update(choices=[])
    eval_src_textbox, eval_tgt_textbox, textbox, src_name_textbox, tgt_name_textbox = "", "", "", "", ""

    if "admin" in worker_id:
        btn = gr.Button.update(visible=True, interactive=True)
        export_worker_id_textbox = gr.Textbox.update(visible=True)
    else:
        btn = gr.Button.update(visible=False, interactive=False)
        export_worker_id_textbox = gr.Textbox.update(visible=False)

    return task_size, example_pointer, total_example_pointer, eval_start, eval_df_all, instance_id_textbox, jump_instance_id_textbox, eval_src_textbox, eval_tgt_textbox, \
           textbox, src_name_textbox, tgt_name_textbox, eval_criteria, btn, export_worker_id_textbox

def pro_eval_data(json_file):
    data = json.load(open(json_file))
    task_to_count = {}
    all_count = 0
    for task in data:
        all_count += len(data[task]['task_info'])
        task_to_count[task] = len(data[task]['task_info'])
    task_to_count['all'] = all_count
    return task_to_count

def build_demo():
    with gr.Blocks(title="CoEval", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()
        task_pointer = gr.State(value=0)
        example_pointer = gr.State(value=0)
        total_example_pointer = gr.State(value=0)
        eval_start = gr.State(value=0)
        eval_df_all = gr.State(value=pd.DataFrame(
                                    data={'Instance ID': [], 'Task information': [], 'Source': [], 'Target': [], 'Criteria': [],
                                          'Final Score': [], 'Evaluation': [],
                                          'APPROVE': [], 'DELETE': [], 'REVISE or ADD': [], 'New Evaluation': []},
                                    columns=['Instance ID', 'Task information', 'Source', 'Target', 'Criteria', 'Final Score', 'Evaluation',
                                             'APPROVE', 'DELETE',
                                             'REVISE or ADD', 'New Evaluation'])
        )
        eval_data_dict = pro_eval_data(args.eval_data_name)
        eval_data_all = gr.State(value=eval_data_dict)
        task_to_eval_data = gr.State(value=json.load(open(args.eval_data_name)))
        all_tasks = gr.State(value=list(json.load(open(args.eval_data_name)).keys()))
        task_size = gr.State(value=eval_data_dict[list(eval_data_dict.keys())[0]])

        gr.Markdown(notice_markdown)
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False).style(container=False)

        with gr.Row(elem_id="annotator_work"):
            with gr.Column(scale=1):
                worker_id_textbox = gr.Textbox(label="Annotator Name", placeholder="Enter your name",
                                               visible=True).style(container=False)
            with gr.Column(min_width=100):
                 choose_task_box = gr.Dropdown(choices=all_tasks.value, value=list(eval_data_dict.keys())[0], label="Evaluated Task", interactive=True)

            with gr.Column(min_width=30):
                 choose_task_btn = gr.Button(value="🆗  OK", interactive=True)  # 我们提前规划好 task_2_example 字典？遍历完所有capabilities，如果这个task评估完了，这个next 无法用了

            with gr.Column(min_width=150):
                annotation_workload = gr.Number(label="Total examples", value=task_size.value, visible=True).style(container=False)
            with gr.Column(min_width=150):
                annotation_finished = gr.Number(label="Finished examples", value=0, visible=True).style(container=False)

            with gr.Column(min_width=30):
                with gr.Row():
                    export_worker_id_textbox = gr.Textbox(label="Export Worker ID", placeholder="Enter Worker ID", visible=False).style(container=False)
                    export_all_btn = gr.Button(value="🔖 Export All", interactive=True, visible=False)  # 我们提前规划好 task_2_example 字典？遍历完所有capabilities，如果这个task评估完了，这个next 无法用了

        gr.Markdown(sep_line)
        gr.Markdown(criteria_evaluation_title_t2)

        with gr.Row():
            with gr.Column(min_width=100):
                instance_id_textbox = gr.Number(label="ID", value=0,
                                      visible=True).style(container=False)  # show_label=False,
            with gr.Column(min_width=100):
                jump_instance_id_textbox = gr.Number(label="Jump to (1-xxx)", value=0,
                                      visible=True).style(container=False)  # show_label=False,
            with gr.Column(min_width=60):
                jump_btn = gr.Button(value="🚀 Jump", interactive=True)  # show_label=False,
            with gr.Column(min_width=350):
                textbox = gr.Textbox(label="Task information", placeholder="Task information",
                                     visible=True).style(container=False)  # show_label=False,
            with gr.Column(min_width=150):
                src_name_textbox = gr.Textbox(label="Evaluated source", placeholder="Task's source field",
                                              visible=True).style(container=False)
            with gr.Column(min_width=150):
                tgt_name_textbox = gr.Textbox(label="Evaluated target", placeholder="Task's target field",
                                              visible=True).style(container=False)

        #########################################################################################################
        with gr.Row():
            with gr.Column(scale=1):
                eval_src_textbox = gr.Textbox(label="Source", placeholder="Instance's source", visible=False).style(
                    container=False)
            with gr.Column(scale=2):
                eval_tgt_textbox = gr.Textbox(label="Target", placeholder="Instance's target", visible=False).style(
                    container=False)

        with gr.Row():
            with gr.Column():
                eval_criteria = gr.Radio(choices=None, label="Evaluated Criteria", interactive=True,
                                         info="Select each criterion to evaluate current instance one by one.") # ["2. Use examples or analogies that are relatable to a five-year-old’s experience.", "1. Use simple and easy-to-understand language."]
        with gr.Row():
            with gr.Column(scale=1):
                eval_check = gr.Button(value="🆗  Confirm Criterion", interactive=True)
            with gr.Column(scale=1):
                final_score = gr.Number(label="Annotator score for this criterion", visible=True).style(
                    container=False)

        eval_output = gr.DataFrame(wrap=True, row_count=(1, "dynamic"), col_count=(5, "fixed"),
                                   label="Evaluation Result",
                                   headers=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕",
                                            "New Evaluation"], interactive=True,
                                   visible=True)  # .style(container=False)

        with gr.Row(visible=False) as eval_button_row:
            eval_next_btn = gr.Button(value="🔄  Next Instance",
                                      interactive=True)
            eval_ok_btn = gr.Button(value="🆗  Save Evaluation", interactive=True)
            eval_last_btn = gr.Button(value="⏮  Last Instance",
                                      interactive=True)
            eval_export_btn = gr.Button(value="🔖 Export", interactive=True)

        eval_csv = gr.File(interactive=False, visible=False)
        eval_progress_csv = gr.File(interactive=False, visible=False)
        eval_progress_eval_csv = gr.File(interactive=False, visible=False)
        eval_error_txt = gr.File(interactive=False, visible=False)
        invoke_mode = gr.State(value=args.invoke_mode)
        # gr.Examples(examples=eval_examples,
        #             inputs=[textbox, src_name_textbox, eval_src_textbox, tgt_name_textbox, eval_tgt_textbox,
        #                     eval_criteria, eval_output, final_score], )

        # gr.Examples(examples=eval_examples, inputs=[textbox_2, eval_src_name_textbox, eval_src_textbox, eval_tgt_name_textbox, eval_tgt_textbox, eval_dropdrown, eval_output, final_score],)
        gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        choose_task_btn.click(fn=choose_cur_task,
                              inputs=[worker_id_textbox, eval_data_all, choose_task_box],
                              outputs=[annotation_workload, example_pointer, total_example_pointer, eval_start, eval_df_all,
                                       instance_id_textbox, jump_instance_id_textbox, eval_src_textbox, eval_tgt_textbox, textbox,
                                       src_name_textbox, tgt_name_textbox, eval_criteria, export_all_btn, export_worker_id_textbox])

        src_name_textbox.change(fn=change_textbox, inputs=src_name_textbox, outputs=eval_src_textbox)
        tgt_name_textbox.change(fn=change_textbox, inputs=tgt_name_textbox, outputs=eval_tgt_textbox)

        eval_export_btn.click(fn=export_eval, inputs=[worker_id_textbox, eval_df_all], outputs=eval_csv)
        export_all_btn.click(fn=export_all, inputs=[export_worker_id_textbox, choose_task_box], outputs=[eval_progress_csv, eval_progress_eval_csv, eval_error_txt])

        eval_ok_btn.click(fn=record_eval_correction,
                          inputs=[eval_output, textbox, src_name_textbox, tgt_name_textbox, eval_src_textbox,
                                  eval_tgt_textbox, eval_criteria, final_score, worker_id_textbox, instance_id_textbox,
                                  task_pointer, example_pointer, total_example_pointer, eval_start, eval_df_all, choose_task_box,],
                          outputs=[eval_output, final_score, eval_df_all])

        eval_check.click(fn=new_eval_criteron,
                             inputs=[eval_criteria, textbox, src_name_textbox, tgt_name_textbox, eval_src_textbox,
                                     eval_tgt_textbox, state, example_pointer, task_pointer, task_to_eval_data, choose_task_box, invoke_mode],
                             outputs=[state, eval_output, eval_start])

        eval_next_btn.click(fn=next_example,
                            inputs=[state, instance_id_textbox, example_pointer, total_example_pointer,
                                    eval_data_all, task_to_eval_data, choose_task_box],
                            outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn,
                                     textbox, src_name_textbox, tgt_name_textbox,
                                     eval_criteria, eval_last_btn, annotation_finished, annotation_workload,
                                     instance_id_textbox, example_pointer, total_example_pointer])

        jump_btn.click(fn=jump_example,
                        inputs=[state, jump_instance_id_textbox, eval_data_all, task_to_eval_data, choose_task_box],
                        outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn, annotation_workload,
                                 annotation_finished, textbox, src_name_textbox, tgt_name_textbox,
                                 eval_criteria, eval_last_btn, instance_id_textbox,
                                 example_pointer, total_example_pointer, eval_output])

        eval_last_btn.click(fn=last_example,
                            inputs=[state, instance_id_textbox, example_pointer, total_example_pointer, eval_data_all, task_to_eval_data, choose_task_box],
                            outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn, annotation_workload,
                                     annotation_finished, textbox, src_name_textbox, tgt_name_textbox,
                                     eval_criteria, eval_last_btn, instance_id_textbox, example_pointer, total_example_pointer, eval_output])

        if args.model_list_mode == "once":
            demo.load(load_demo, [url_params],
                      [state, model_selector, eval_output,
                       textbox, src_name_textbox, tgt_name_textbox, eval_src_textbox, eval_tgt_textbox,
                       eval_button_row, task_pointer, example_pointer, total_example_pointer, eval_start, eval_df_all,
                       eval_data_all, task_to_eval_data, all_tasks],
                      _js=get_window_url_params)
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None,
                      [state, model_selector, eval_output,
                       textbox, src_name_textbox, tgt_name_textbox, eval_src_textbox, eval_tgt_textbox,
                       eval_button_row, task_pointer, example_pointer, total_example_pointer, eval_start, eval_df_all,
                       eval_data_all, task_to_eval_data, all_tasks])
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo

if __name__ == "__main__":
    # todo: save annotation history
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)  # , default=7860
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=20)  # 10
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--eval-data-name", type=str, default="combine_data.json")
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--invoke-mode", type=str, default="offline or online")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true", help="Enable content moderation")
    parser.add_argument("--llm_prepare", action="store_true", help="prepare llm's evaluation in advance")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()
    logger.info(args)
    demo = build_demo()

    demo.queue(
        concurrency_count=args.concurrency_count,
        status_update_rate=10,
        api_open=False
    )
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
    )

