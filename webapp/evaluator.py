import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """Different separator style."""
    # Automatically assign the integer value to the values of enum class attributes.
    SINGLE = auto()
    TWO = auto()
# [<SeparatorStyle.SINGLE: 1>, <SeparatorStyle.TWO: 2>]

def firstletterupper(text):
    return text[0].upper() + text[1:]


@dataclasses.dataclass
class Collaboration:
    """A class that keeps all collaboration history."""
    system: str  # description about system
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None



    def combine_cap_query(self, task_info, eval_tgt):
        # The task is to provide answers to Reddit forum questions which are comprehensible by five year olds. What capabilities of the answers should there be? List directly.
        ret = ' '.join([task_info, "What capabilities of the", eval_tgt, "should there be? List directly."])
        ret = firstletterupper(ret)
        return ret


    def combine_eval_query(self, task_info, src_name, src, tgt_name, tgt, eval_cap):
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
        # print('combine_eval_query: ')
        # print(final)
        return final

    def get_prompt(self):
        # prompt in chat format
        # self.system....###Human:xxx....###Assistant:xxxx.....
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_prompt_eval(self):
        '''
        evaluation prompt todo: need to revise
        '''
        ret = ""
        for role, message in self.messages:
            if message:
                ret += message
        print('evaluation ret: ', ret)
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])


    def invalid_input(self, msg):
        self.invalid_msg = msg


    def to_gradio_chatbot(self):
        # [['human text', 'assistant text'], ['human text', 'assistant text']]
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:  # 说话者1
                ret.append([msg, None])
            else:  # 说话者2，用msg填充上一个none
                ret[-1][-1] = msg
        return ret


    def copy(self):
        return Collaboration(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


task_to_prompt_capabilities = {
    'eli5': {'setting': "You are a helpful evaluation expert in NLP.",
             'prompt_task': "The task is to provide answers to Reddit forum questions which are comprehensible by five year olds.",
             'prompt_capability': "What capabilities of the answers should there be? List directly.",
             'received_capabilities': "",
             'capability': "The answer should use simple and clear language that is easy for a five-year-old to understand.\n\n",
             'prompt_evaluation_task': "You will be given an answer for a Reddit forum question which is expected to be comprehensible by five year olds. Your task is to evaluate the summary based on a premise.\n\n",
             'evaluation_steps': "Evaluation Steps: \n1. Evaluate whether this answer satisfy the premise. Give a conclusion.\n2. Assign a score for this answer on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the premise.\n3. List evidence by quoting sentences of the answer to support your conclusion.\n",
             'received_evaluation': "",
}
}

collab_v1 = Collaboration(
    system="A chat between a human evaluator and an intelligence evaluation assistant. "
           "The assistant gives helpful, detailed, and expert answers to the human's questions.",  # context
    roles=("Human", "Assistant"),
    messages=(  # (role, message), (role, message)
        # ("Human", 'hello, hhahah'),
        # ("Assistant", 'byebye'),
        # ("Human", 'why??'),
        # ("Assistant", 'don\'t ask me!'),
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

collab_v2 = Collaboration(
    system="",  # context
    roles=("Human", "Assistant"),
    messages=(  # (role, message), (role, message)
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


default_collaboration = collab_v2
collab_templates = {
    "v1": collab_v1,
    "v2": collab_v2
}



if __name__ == "__main__":
    print(default_collaboration.get_prompt())
    print(collab_v1.to_gradio_chatbot())