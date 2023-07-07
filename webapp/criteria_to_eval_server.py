import copy
import os
import argparse
import pdb
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
from gradio_content import priority, css, notice_markdown, criteria_query_title, criteria_evaluation_title, learn_more_markdown, sep_line, get_window_url_params, cap_examples, eval_examples


logger = build_logger("criteria_to_eval", "criteria_to_eval.log")
headers = {"User-Agent": "fastchat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

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
    example_pointer = 0
    total_example_pointer = 0
    eval_df_all = pd.DataFrame(
        data={'Instance ID': [], 'Task information': [], 'Source': [], 'Target': [], 'Criteria': [],
              'Final Score': [], 'Evaluation': [],
              'APPROVE': [], 'DELETE': [], 'REVISE or ADD': [], 'New Evaluation': []},
        columns=['Instance ID', 'Task information', 'Source', 'Target', 'Criteria', 'Final Score', 'Evaluation',
                 'APPROVE', 'DELETE',
                 'REVISE or ADD', 'New Evaluation'])
    eval_data_all, task_to_eval_data, all_tasks = read_back_file()

    cap_start = 0
    eval_start = 0

    state = default_collaboration.copy()
    return (state,
            gr.DataFrame.update(visible=True),
            gr.DataFrame.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Row.update(visible=True),
            gr.Row.update(visible=True),
            example_pointer, total_example_pointer, eval_start, cap_start,
            eval_df_all, eval_data_all, task_to_eval_data, all_tasks)



def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    example_pointer = 0
    total_example_pointer = 0
    eval_start = 0
    cap_start = 0
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
            gr.DataFrame.update(visible=True),
            gr.DataFrame.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Row.update(visible=True),
            gr.Row.update(visible=True),
            example_pointer, total_example_pointer, eval_start, cap_start,
            eval_df_all, eval_data_all, task_to_eval_data, all_tasks
            )


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_collaboration.copy()
    return (state,
            pd.DataFrame({"Criteria": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Criteria": [""]},
                         columns=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"]),
            "",
            "",
            "") + (enable_btn,) * 3


def clear_eval_history(request: gr.Request):
    # state, eval_output, eval_src_textbox, eval_tgt_textbox, final_score
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_collaboration.copy()
    return (state,
            pd.DataFrame({"Evaluation": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]}, columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"]),
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

def next_task(state, task_to_eval_data, choose_task_box):
    example_pointer = 0
    cur_task_data = task_to_eval_data.value[choose_task_box]
    task_info = cur_task_data['task_info'][example_pointer]
    source_name = cur_task_data['source_name'][example_pointer]
    target_name = cur_task_data['target_name'][example_pointer]

    new_text = state.combine_cap_query(task_info, target_name)

    template_name = "v2"
    new_state = collab_templates[template_name].copy()
    new_state.conv_id = uuid.uuid4().hex
    new_state.append_message(new_state.roles[0], new_text)
    new_state.append_message(new_state.roles[1], None)
    state = new_state

    state.skip_next = False
    empty_cap_df = pd.DataFrame({"Criteria": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Criteria": [""]},
                         columns=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"])
    return state, empty_cap_df, task_info, source_name, target_name, enable_btn, example_pointer  # enable_btn: eval_next_btn


def http_bot_cap(state, temperature, max_new_tokens):
    '''
    then(http_bot, inputs=[state, model_selector, temperature, max_output_tokens],
                   outputs=[state, cap_output] + btn_list)
    '''
    template_name = "v2"
    new_state = collab_templates[template_name].copy()
    new_state.conv_id = uuid.uuid4().hex
    new_state.append_message(new_state.roles[0], state.messages[-2][1])
    new_state.append_message(new_state.roles[1], None)
    state = new_state

    # Query worker address
    model_name = args.model_name
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, pd.DataFrame({"Criteria": ["No available worker!"],
                                    "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Criteria": [""]},
                                   columns=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"]), 0,
              enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt_eval()

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
    }
    # logger.info(f"http_bot_cap ==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True, timeout=100)
        lt = time.localtime(time.time())
        to_time = time.strftime("%Y-%m-%d %H:%M:%S", lt)
        cap_start = to_time
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:  # success
                    output = data["text"].strip()
                    output, output_df = post_process_code(output, cap_or_eval='cap')
                    state.messages[-1][-1] = output + "▌"
                    yield (state, output_df, cap_start, no_change_btn)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state,
                           pd.DataFrame({"Criteria": [f" (error_code: {data['error_code']})"],
                                         "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""],
                                         "New Criteria": [""]},
                                        columns=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"]), cap_start, enable_btn)
                    return
                time.sleep(0.02)
    except requests.exceptions.RequestException as e:
        # usually vpn -> connection issue
        state.messages[-1][-1] = server_error_msg + f" (error_code: 4)"
        yield (state,
               pd.DataFrame({"Criteria": [f" (error_code: 4)"],
                             "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""],
                             "New Criteria": [""]},
                            columns=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"]), 0, enable_btn)
        return
    state.messages[-1][-1] = state.messages[-1][-1][:-1]


def change_textbox(change_text):
    return gr.update(visible=True, label=change_text)


def record_cap_correction(correct_cap, task_info, src_name, tgt_name, workerid, choose_task_box, example_pointer, total_example_pointer, instance_id, cap_start, cap_df_all, task_to_eval_data, all_tasks):
    lt = time.localtime(time.time())
    to_time = time.strftime("%Y-%m-%d %H:%M:%S", lt)
    cap_end = to_time

    ret = correct_cap.to_json(orient="values")
    parsed = json.loads(ret)
    parsed.insert(0, {'worker_id': workerid, 'task_info': task_info, 'src_name': src_name, 'tgt_name': tgt_name,
                      'task_name': choose_task_box, 'cap_start_time': cap_start, 'cap_end_time': cap_end})
    ret = json.dumps(parsed)

    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-{workerid}-criteria-correction.json")
    with open(name, "a") as fout:
        fout.write(ret + "\n")

    # update corrected criteria and eval_textbox
    new_correct_cap = copy.copy(correct_cap)
    new_correct_cap = new_correct_cap[new_correct_cap["DELETE ❎"] == ""]  # str.contains  .startswith("delete") == False
    new_correct_cap['New Criteria'] = np.where(new_correct_cap['APPROVE ✅'] != "", new_correct_cap["Criteria"], new_correct_cap["New Criteria"])
    new_correct_cap = new_correct_cap[["New Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "Criteria"]]


    # begin step 2, this task's first instance
    example_pointer = 0
    cur_task_data = task_to_eval_data.value[choose_task_box]
    source = cur_task_data['source'][example_pointer]
    target = cur_task_data['target'][example_pointer]

    example_pointer += 1
    total_example_pointer += 1
    instance_id += 1

    # this task's first instance's first criteron
    criteria_choices = new_correct_cap["New Criteria"].values.tolist()

    this_task_info = [task_info] + [""] * (len(correct_cap)-1)
    this_source = [src_name] + [""] * (len(correct_cap)-1)
    this_target = [tgt_name] + [""] * (len(correct_cap)-1)

    for idx in range(len(correct_cap)):
        cap_df_all.loc[len(cap_df_all.index)] = [this_task_info[idx], this_source[idx], this_target[idx], correct_cap['Criteria'][idx],
                                                 correct_cap['APPROVE ✅'][idx], correct_cap['DELETE ❎'][idx], correct_cap['REVISE 🔀 or ADD 🆕'][idx], correct_cap['New Criteria'][idx]]

    return gr.Radio.update(choices=criteria_choices, value=criteria_choices[0]), source, target, disable_btn, example_pointer, total_example_pointer, instance_id


def next_example(state, instance_id_textbox, cap_output, eval_data_all, example_pointer, total_example_pointer, task_to_eval_data, choose_task_box):
    '''
        eval_next_btn.click(fn=next_example,
                            inputs=[state, instance_id_textbox, cap_output, eval_data_all
                                    example_pointer, total_example_pointer, task_to_eval_data, choose_task_box],
                            outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn,
                                     annotation_workload, annotation_finished, eval_criteria, eval_last_btn])
    )
    '''
    cur_task_data = task_to_eval_data.value[choose_task_box]
    if example_pointer == len(cur_task_data['source']):
        source = ""
        target = ""
        btn = disable_btn  # next button
        cap_choices = [None]
    else:
        source = cur_task_data['source'][example_pointer]
        target = cur_task_data['target'][example_pointer]

        new_correct_cap = copy.copy(cap_output)
        new_correct_cap = new_correct_cap[new_correct_cap["DELETE ❎"] == ""]  # str.contains  .startswith("delete") == False
        new_correct_cap['New Criteria'] = np.where(new_correct_cap['APPROVE ✅'] != "", new_correct_cap["Criteria"], new_correct_cap["New Criteria"])
        cap_choices = new_correct_cap["New Criteria"].values.tolist()

        btn = enable_btn
        example_pointer += 1
        total_example_pointer += 1

    if example_pointer == 1:
        last_btn = disable_btn
    else:
        last_btn = enable_btn

    instance_id_textbox += 1

    annotation_finished = (example_pointer - 1)
    annotation_workload = (eval_data_all.value[choose_task_box] - (example_pointer - 1))
    return state, source, target, btn, \
           annotation_workload, annotation_finished, \
           gr.Radio.update(value=cap_choices[0], choices=cap_choices), last_btn, \
           instance_id_textbox, example_pointer, total_example_pointer


def new_eval_criteron(criteria, task_info, src_name, tgt_name, src, tgt, state):
    new_text = state.combine_eval_query(task_info, src_name, src, tgt_name, tgt, criteria)

    template_name = "v2"
    new_state = collab_templates[template_name].copy()
    new_state.conv_id = uuid.uuid4().hex
    new_state.append_message(new_state.roles[0], new_text)
    new_state.append_message(new_state.roles[1], None)
    state = new_state
    state.skip_next = False

    return state


def http_bot_eval(state, temperature, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    model_name = args.model_name

    template_name = "v2"
    new_state = collab_templates[template_name].copy()
    new_state.conv_id = uuid.uuid4().hex
    new_state.append_message(new_state.roles[0], state.messages[-2][1])
    new_state.append_message(new_state.roles[1], None)
    state = new_state

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
    # logger.info(f"==== request ====\n{pload}")
    state.messages[-1][-1] = "▌"

    try:
        response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True, timeout=100)
        lt = time.localtime(time.time())
        to_time = time.strftime("%Y-%m-%d %H:%M:%S", lt)
        eval_start = to_time

        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:  # success
                    output = data["text"].strip()
                    output, output_df = post_process_code(output, cap_or_eval='eval')
                    state.messages[-1][-1] = output + "▌"
                    yield (state, output_df, eval_start) + (no_change_btn,) * 2
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state,
                           pd.DataFrame({"Evaluation": [f" (error_code: {data['error_code']})"], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]}, columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "Mew Evaluation"]), 0) \
                          + (enable_btn,) * 2
                    return
                time.sleep(0.02)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg + f" (error_code: 4)"
        yield (state, pd.DataFrame({"Evaluation": [f" (error_code: 4)"], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]}, columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"]), 0) \
              + (enable_btn,) * 2
        return
    state.messages[-1][-1] = state.messages[-1][-1][:-1]


def record_eval_correction(eval_result, task_info, src_name, tgt_name, eval_src, eval_tgt, eval_cap, final_score, workerid,
                           choose_task_box, instance_id, example_pointer, total_example_pointer, eval_start, eval_df_all):

    lt = time.localtime(time.time())
    to_time = time.strftime("%Y-%m-%d %H:%M:%S", lt)
    eval_end = to_time

    ret = eval_result.to_json(orient="values")
    parsed = json.loads(ret)
    parsed.insert(0, {'worker_id': workerid, 'task_info': task_info, 'src_name': src_name, 'tgt_name': tgt_name,
                      'eval_src': eval_src, 'eval_tgt': eval_tgt, 'eval_cap': eval_cap, 'final_score': final_score,
                      'task_name': choose_task_box, 'task_example_pointer': example_pointer,
                      'example_pointer': total_example_pointer, 'eval_start_time': eval_start,
                      'eval_end_time': eval_end})
    ret = json.dumps(parsed)

    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-{workerid}-evaluation-correction.json")
    with open(name, "a") as fout:
        fout.write(ret + "\n")

    empty_df = pd.DataFrame({"Evaluation": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]},
                     columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"])


    this_task_info = [task_info] + [""] * (len(eval_result)-1)
    this_source = [src_name] + [""] * (len(eval_result)-1)
    this_target = [tgt_name] + [""] * (len(eval_result)-1)
    this_criteron = [eval_cap] + [""] * (len(eval_result)-1)
    this_final_score = [final_score] + [""] * (len(eval_result)-1)
    this_instance_id = [instance_id] + [""] * (len(eval_result) - 1)

    for idx in range(len(eval_result)):
        new_row = [this_instance_id[idx], this_task_info[idx],
                   this_source[idx], this_target[idx],
                   this_criteron[idx], this_final_score[idx],
                   eval_result['Evaluation'][idx],
                   eval_result['APPROVE ✅'][idx], eval_result['DELETE ❎'][idx],
                   eval_result['REVISE 🔀 or ADD 🆕'][idx],
                   eval_result['New Evaluation'][idx]]
        eval_df_all.loc[len(eval_df_all.index)] = new_row


    return empty_df, gr.Number.update(value=None), eval_df_all



def last_example(state, cap_output, instance_id_textbox, example_pointer, total_example_pointer, task_to_eval_data, choose_task_box, eval_data_all):
    example_pointer -= 2
    total_example_pointer -= 2

    cur_task_data = task_to_eval_data.value[choose_task_box]
    source = cur_task_data['source'][example_pointer]
    target = cur_task_data['target'][example_pointer]
    btn = enable_btn

    new_correct_cap = copy.copy(cap_output)
    new_correct_cap = new_correct_cap[new_correct_cap["DELETE ❎"] == ""]  # str.contains  .startswith("delete") == False
    new_correct_cap['New Criteria'] = np.where(new_correct_cap['APPROVE ✅'] != "", new_correct_cap["Criteria"],
                                               new_correct_cap["New Criteria"])
    cap_choices = new_correct_cap["New Criteria"].values.tolist()

    example_pointer += 1
    total_example_pointer += 1
    instance_id_textbox -= 1

    if example_pointer == 1:
        last_btn = disable_btn
    else:
        last_btn = enable_btn

    empty_df = pd.DataFrame(
        {"Evaluation": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Evaluation": [""]},
        columns=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"])

    return state, source, target, btn, (eval_data_all.value[choose_task_box] - total_example_pointer), total_example_pointer, \
           gr.Radio.update(value=cap_choices[0], choices=cap_choices), last_btn, instance_id_textbox, \
           example_pointer, total_example_pointer, empty_df


def jump_example(state, instance_id, eval_data_all, task_to_eval_data, choose_task_box, cap_output):
    example_pointer = (int(instance_id)-1)  # todo index and no.
    total_example_pointer = (int(instance_id)-1)

    cur_task_data = task_to_eval_data.value[choose_task_box]
    source = cur_task_data['source'][example_pointer]
    target = cur_task_data['target'][example_pointer]

    new_correct_cap = copy.copy(cap_output)
    new_correct_cap = new_correct_cap[new_correct_cap["DELETE ❎"] == ""]  # str.contains  .startswith("delete") == False
    new_correct_cap['New Criteria'] = np.where(new_correct_cap['APPROVE ✅'] != "", new_correct_cap["Criteria"],
                                               new_correct_cap["New Criteria"])
    cap_choices = new_correct_cap["New Criteria"].values.tolist()

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
           gr.Radio.update(value=cap_choices[0], choices=cap_choices), last_btn, cur_instance_id, \
           example_pointer, total_example_pointer, empty_df



def change(cnt):
    data = [[1, 2], [3,4]]
    return data

def clear():
    data = pd.DataFrame(data={"A": [""], "B": [""]}, columns=["A", "B"])
    return (data, data)


def export_cap(worker_id, cap_df_all):
    t = datetime.datetime.now()
    random_string = ''.join(random.choices(string.ascii_letters, k=6))
    if os.path.exists('annotation_data/annotator_export/') is False:
        os.makedirs('annotation_data/annotator_export/')

    cap_df_all.to_excel(f"annotation_data/annotator_export/{t.year}-{t.month:02d}-{t.day:02d}-{worker_id}_criteria_{random_string}.xlsx", index=None, header=True)
    return gr.File.update(value=f"annotation_data/annotator_export/{t.year}-{t.month:02d}-{t.day:02d}-{worker_id}_criteria_{random_string}.xlsx",
                          visible=True)

def export_eval(worker_id, eval_df_all):
    t = datetime.datetime.now()
    random_string = ''.join(random.choices(string.ascii_letters, k=6))
    eval_df_all.to_excel(f"annotation_data/annotator_export/{t.year}-{t.month:02d}-{t.day:02d}-{worker_id}_evaluation_{random_string}.xlsx", index=None, header=True)
    return gr.File.update(value=f"annotation_data/annotator_export/{t.year}-{t.month:02d}-{t.day:02d}-{worker_id}_evaluation_{random_string}.xlsx",
                          visible=True)


def pro_eval_data(json_file):
    data = json.load(open(json_file))
    task_to_count = {}
    all_count = 0
    for task in data:
        all_count += len(data[task]['task_info'])
        task_to_count[task] = len(data[task]['task_info'])
    task_to_count['all'] = all_count
    return task_to_count


def choose_cur_task(state, worker_id, eval_data_all, choose_task_box, task_to_eval_data):
    task_size = eval_data_all.value[choose_task_box]

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
    # eval_src_textbox, eval_tgt_textbox, textbox, src_name_textbox, tgt_name_textbox = "", "", "", "", ""

    if "admin" in worker_id:
        btn = gr.Button.update(visible=True, interactive=True)
        export_worker_id_textbox = gr.Textbox.update(visible=True)
    else:
        btn = gr.Button.update(visible=False, interactive=False)
        export_worker_id_textbox = gr.Textbox.update(visible=False)

    cur_task_data = task_to_eval_data.value[choose_task_box]
    task_info = cur_task_data['task_info'][example_pointer]
    source_name = cur_task_data['source_name'][example_pointer]
    target_name = cur_task_data['target_name'][example_pointer]
    new_text = state.combine_cap_query(task_info, target_name)

    template_name = "v2"
    new_state = collab_templates[template_name].copy()
    new_state.conv_id = uuid.uuid4().hex
    new_state.append_message(new_state.roles[0], new_text)
    new_state.append_message(new_state.roles[1], None)
    state = new_state

    state.skip_next = False
    empty_cap_df = pd.DataFrame(
        {"Criteria": [""], "APPROVE ✅": [""], "DELETE ❎": [""], "REVISE 🔀 or ADD 🆕": [""], "New Criteria": [""]},
        columns=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"])

    return state, task_size, example_pointer, total_example_pointer, eval_start, eval_df_all, instance_id_textbox, jump_instance_id_textbox, \
           task_info, source_name, target_name, eval_criteria, btn, export_worker_id_textbox, empty_cap_df


def build_demo():
    with gr.Blocks(title="CoEval", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()
        example_pointer = gr.State(value=0)
        total_example_pointer = gr.State(value=0)
        cap_start = gr.State(value=0)
        eval_start = gr.State(value=0)
        cap_df_all = gr.State(value=pd.DataFrame(
            data={'Task information': [], 'Source': [], 'Target': [], 'Criteria': [], 'APPROVE': [], 'DELETE': [],
                  'REVISE or ADD': [], 'New Criteria': []},
            columns=['Task information', 'Source', 'Target', 'Criteria', 'APPROVE', 'DELETE', 'REVISE or ADD',
                     'New Criteria']))
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
        task_to_eval_data = gr.State(value=json.load(open(args.eval_data_name)))  # 'qualification_v1_combine.json'
        all_tasks = gr.State(value=list(json.load(open(args.eval_data_name)).keys()))
        task_size = gr.State(value=eval_data_dict[list(eval_data_dict.keys())[0]])

        gr.Markdown(notice_markdown)

        with gr.Row(elem_id="annotator_work"):
            with gr.Column(scale=1):
                worker_id_textbox = gr.Textbox(label="Annotator Name", placeholder="Enter your name",
                                               visible=True).style(container=False)
            with gr.Column(min_width=100):
                choose_task_box = gr.Dropdown(choices=all_tasks.value, value=list(eval_data_dict.keys())[0], label="Evaluated Task", interactive=True)

            with gr.Column(min_width=30):
                choose_task_btn = gr.Button(value="🆗  OK", interactive=True)  # 我们提前规划好 task_2_example 字典？遍历完所有capabilities，如果这个task评估完了，这个next 无法用了

            with gr.Column(min_width=200):
                annotation_workload = gr.Number(label="Total examples", value=task_size.value, visible=True).style(container=False)
            with gr.Column(min_width=200):
                annotation_finished = gr.Number(label="Finished examples", value=0, visible=True).style(container=False)

            with gr.Column(min_width=30):
                with gr.Row():
                    export_worker_id_textbox = gr.Textbox(label="Export Worker ID", placeholder="Enter Worker ID", visible=False).style(container=False)
                    export_all_btn = gr.Button(value="🔖 Export All", interactive=True, visible=False)  # 我们提前规划好 task_2_example 字典？遍历完所有capabilities，如果这个task评估完了，这个next 无法用了


        gr.Markdown(sep_line)
        gr.Markdown(criteria_query_title)

        with gr.Row():
            with gr.Column(scale=10):
                textbox = gr.Textbox(label="Task information", placeholder="Enter task information", visible=False).style(container=False)  # show_label=False,
            with gr.Column(scale=1):
                src_name_textbox = gr.Textbox(label="Evaluated source", placeholder="Enter task's source name", visible=False).style(container=False)
            with gr.Column(scale=1):
                tgt_name_textbox = gr.Textbox(label="Evaluated target", placeholder="Enter task's target name", visible=False).style(container=False)

        cap_output = gr.DataFrame(wrap=True,
                                  row_count=(1, "dynamic"), col_count=(5, "fixed"),
                                  label="Evaluated Criteria",
                                  headers=["Criteria", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Criteria"],
                                  interactive=True)  # .style(container=False)

        with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0, step=0.1, interactive=True, label="Temperature",)
            max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=256, step=64, interactive=True, label="Max output tokens",)

        with gr.Row(visible=False) as button_row:
            cap_ok_btn = gr.Button(value="🆗  Save Criteria", interactive=True)
            cap_export_btn = gr.Button(value="🔖 Export", interactive=True)
        # gr.Examples(examples=cap_examples, inputs=[textbox, src_name_textbox, tgt_name_textbox, cap_output],)
        cap_csv = gr.File(interactive=False, visible=False)

        #########################################################################################################
        gr.Markdown(sep_line)
        gr.Markdown(criteria_evaluation_title)

        with gr.Row():
            with gr.Column(min_width=100):
                instance_id_textbox = gr.Number(label="Instance ID", value=0,
                                      visible=True).style(container=False)  # show_label=False,
            with gr.Column(min_width=100):
                jump_instance_id_textbox = gr.Number(label="Jump to (1-xxx)", value=0,
                                      visible=True).style(container=False)  # show_label=False,
            with gr.Column(min_width=60):
                jump_btn = gr.Button(value="🚀 Jump", interactive=True)  # show_label=False,

        with gr.Row():
            with gr.Column(min_width=150):
                eval_src_textbox = gr.Textbox(label="Source", placeholder="Instance's source", visible=False).style(
                    container=False)
            with gr.Column(min_width=150):
                eval_tgt_textbox = gr.Textbox(label="Target", placeholder="Instance's target", visible=False).style(
                    container=False)

        with gr.Row():
            with gr.Column():
                eval_criteria = gr.Radio(choices=None, label="Evaluated Criteria", interactive=True,
                                         info="Select each criterion to evaluate current instance one by one.") # ["2. Use examples or analogies that are relatable to a five-year-old’s experience.", "1. Use simple and easy-to-understand language."]
            with gr.Column(scale=1):
                eval_check = gr.Button(value="🆗  Confirm Criterion", interactive=True)
                final_score = gr.Number(label="Annotator score for this criterion", visible=True).style(
                    container=False)

        eval_output = gr.DataFrame(wrap=True, row_count=(1, "dynamic"), col_count=(5, "fixed"),
                                   label="Evaluation Result",
                                   headers=["Evaluation", "APPROVE ✅", "DELETE ❎", "REVISE 🔀 or ADD 🆕", "New Evaluation"], interactive=True, visible=True)  # .style(container=False)

        with gr.Row(visible=False) as eval_button_row:
            eval_next_btn = gr.Button(value="🔄  Next Instance", interactive=True)  # 我们提前规划好 task_2_example 字典？遍历完所有capabilities，如果这个task评估完了，这个next 无法用了
            eval_ok_btn = gr.Button(value="🆗  Save Evaluation", interactive=True)
            eval_last_btn = gr.Button(value="⏮  Last Instance", interactive=True)  # 我们提前规划好 task_2_example 字典？遍历完所有capabilities，如果这个task评估完了，这个next 无法用了
            eval_export_btn = gr.Button(value="🔖 Export", interactive=True)
        eval_csv = gr.File(interactive=False, visible=False)

        # gr.Examples(examples=eval_examples, inputs=[textbox_2, eval_src_name_textbox, eval_src_textbox, eval_tgt_name_textbox, eval_tgt_textbox, eval_dropdrown, eval_output, final_score],)
        gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        src_name_textbox.change(fn=change_textbox, inputs=src_name_textbox, outputs=eval_src_textbox)
        tgt_name_textbox.change(fn=change_textbox, inputs=tgt_name_textbox, outputs=eval_tgt_textbox)

        cap_export_btn.click(fn=export_cap, inputs=[worker_id_textbox, cap_df_all], outputs=cap_csv)
        eval_export_btn.click(fn=export_eval, inputs=[worker_id_textbox, eval_df_all], outputs=eval_csv)

        choose_task_btn.click(fn=choose_cur_task,
                              inputs=[state, worker_id_textbox, eval_data_all, choose_task_box, task_to_eval_data],
                              outputs=[state, annotation_workload, example_pointer, total_example_pointer, eval_start, eval_df_all,
                                       instance_id_textbox, jump_instance_id_textbox, textbox, src_name_textbox, tgt_name_textbox,
                                       eval_criteria, export_all_btn, export_worker_id_textbox, cap_output]
        ).then(http_bot_cap,
               inputs=[state, temperature, max_output_tokens],
               outputs=[state, cap_output, cap_start, cap_ok_btn])


        # once clicking cap_ok_btn，we will display corrected evaluation criteria
        cap_ok_btn.click(fn=record_cap_correction,
                         inputs=[cap_output, textbox, src_name_textbox, tgt_name_textbox, worker_id_textbox,
                                 choose_task_box, example_pointer, total_example_pointer, instance_id_textbox, cap_start, cap_df_all, task_to_eval_data, all_tasks],
                         outputs=[eval_criteria, eval_src_textbox, eval_tgt_textbox, eval_last_btn, example_pointer, total_example_pointer, instance_id_textbox])


        eval_next_btn.click(fn=next_example,
                            inputs=[state, instance_id_textbox, cap_output, eval_data_all,
                                    example_pointer, total_example_pointer, task_to_eval_data, choose_task_box],
                            outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn,
                                     annotation_workload, annotation_finished, eval_criteria, eval_last_btn,
                                     instance_id_textbox, example_pointer, total_example_pointer])

        eval_last_btn.click(fn=last_example,
                            inputs=[state, cap_output, instance_id_textbox, example_pointer, total_example_pointer,
                                    task_to_eval_data, choose_task_box, eval_data_all],
                            outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn, annotation_workload,
                                     annotation_finished, eval_criteria, eval_last_btn, instance_id_textbox,
                                     example_pointer, total_example_pointer, eval_output])

        jump_btn.click(fn=jump_example,
                        inputs=[state, jump_instance_id_textbox, eval_data_all, task_to_eval_data, choose_task_box, cap_output],
                        outputs=[state, eval_src_textbox, eval_tgt_textbox, eval_next_btn, annotation_workload,
                                 annotation_finished, eval_criteria, eval_last_btn, instance_id_textbox,
                                 example_pointer, total_example_pointer, eval_output])

        eval_check.click(fn=new_eval_criteron,
                             inputs=[eval_criteria, textbox, src_name_textbox, tgt_name_textbox, eval_src_textbox, eval_tgt_textbox, state],
                             outputs=[state]
        ).then(http_bot_eval,
               inputs=[state, temperature, max_output_tokens],
               outputs=[state, eval_output, eval_start, eval_next_btn, eval_ok_btn])

        eval_ok_btn.click(fn=record_eval_correction,
                          inputs=[eval_output, textbox, src_name_textbox, tgt_name_textbox, eval_src_textbox, eval_tgt_textbox, eval_criteria, final_score, worker_id_textbox,
                                  choose_task_box, instance_id_textbox, example_pointer, total_example_pointer, eval_start, eval_df_all],
                          outputs=[eval_output, final_score, eval_df_all])


        if args.model_list_mode == "once":
            demo.load(load_demo, [url_params],
                      [state, cap_output, eval_output,
                       textbox, src_name_textbox, tgt_name_textbox, eval_src_textbox, eval_tgt_textbox,
                       button_row, eval_button_row, example_pointer, total_example_pointer, eval_start, cap_start,
                       eval_df_all, eval_data_all, task_to_eval_data, all_tasks],
                      _js=get_window_url_params)
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None,
                      [state, cap_output, eval_output,
                       textbox, src_name_textbox, tgt_name_textbox, eval_src_textbox, eval_tgt_textbox,
                       button_row, eval_button_row, example_pointer, total_example_pointer, eval_start, cap_start,
                       eval_df_all, eval_data_all, task_to_eval_data, all_tasks])
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", default='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')

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)  # , default=7860
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=20)  # 10
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--invoke-mode", type=str, default="offline or online")
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--eval-data-name", type=str, default="combine_data.json")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true", help="Enable content moderation")
    parser.add_argument("--llm_prepare", action="store_true", help="prepare llm's evaluation in advance")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()
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
