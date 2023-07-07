import pandas as pd

priority = {
    "gpt-3.5-turbo-0301": "aaaaaaa",
    "gpt-3.5-turbo": "aaaaaab",
}

# evaluation_text = "1. The answer partially satisfies the premise. While it does use simple language, it includes some complex ideas and terminology that may not be easily understood by a five-year-old. Additionally, it does not directly answer the question of how diet soda can make someone gain weight.\n"\
#        "2. I would assign a score of 3 out of 5 based on the premise.\n"\
#        "3. Evidence to support the conclusion includes:\n"\
#        "- The answer uses simple language, such as \"regular Coca-Cola\" and \"soft drinks.\"\n"\
#        "- However, it also includes complex ideas and terminology, such as \"controlled setting\" and \"calorie consumption.\"\n"
# "- The answer does not directly answer the question of how diet soda can make someone gain weight, but instead discusses the effects of regular soda on calorie consumption and sugar intake."

evaluation_text = "1. The answer satisfies the premise as it uses relatable examples and analogies that a five-year-old can understand.\n"\
                  "2. I would assign a score of 4 to this answer as it effectively uses relatable examples but could have been more concise and straightforward.\n"\
                  "3. Evidence:\n"\
                  "- \"websites like Reddit would have to find a new way to share data with each other\" - This sentence uses relatable language and explains the potential consequence of Net Neutrality going away in a way that a five-year-old can understand."


# df = pd.DataFrame(
#     data={
#         'evaluation': evaluation_text.split('\n'),
#         "correction": ["pass"] + ["" for _ in range(len(evaluation_text.split('\n'))-1)],
#         "final evaluation": ["" for _ in range(len(evaluation_text.split('\n')))]}
# )
# eval_examples = [["The task is to provide an answer to a Reddit forum question that is comprehensible by five-year-olds.",
#                    "reddit question",
#                   "How can diet soda make me gain weight?",
#                   "answer",
#                   "A study from San Diego State University. In a controlled setting, researchers found that drinking two cans of regular Coca-Cola every day for three days caused people to consume almost 300 extra calories above their normal daily intake without significantly affecting how much they weighed.[2] At this point it’s important you don't get too excited about Coke and start stockpiling the pantry with them; while hundreds or even thousands might not seem like such an awful lot at first glance (hey—it is just 3 sodas right?) these small increases add up quick if consumed regularly over time! In fact those few soft drinks do more than simply affect your body's calorie consumption: They also contain loads on sugar[3], which as we know by now has been identified as one major factor contributing obesity in America today![4]. So instead think twice before reaching into any pop machines near work when looking around lunchtime – doing so once may be no big deal but after years spent slowly consuming excess amounts each year keep pilling onto our waistlines regardless where being overweight really hurts us most...right under chin.",
#                   "The answer should use simple and clear language that is easy for a five-year-old to understand.",
#                   df,
#                   "3"
#                 ],]


df = pd.DataFrame(
    data={
        'evaluation': evaluation_text.split('\n'),
        "APPROVE ✅": ["", "", 1, ""],
        "DELETE ❎": [""] * 4,
        "REVISE 🔀 or ADD 🆕": ["no examples", "no examples", "", "unrelated evaluation"],
        "New Evaluation": ["1. The answer doesn’t satisfy the premise as it doesn’t use any relatable examples or analogies that a five-year-old can understand.",
                           "2. I would assign a score of 0 to this answer as it fails to give any examples or analogies.",
                           "",
                           "- \"If Net Neutrality goes away, websites like Reddit would have to find a new way to share data with each other.\" – The whole text doesn’t use any examples or analogies."]
    }
)



evaluation_text1 =  "1. The answer does not satisfy the premise of using simple and easy-to-understand language. The answer uses technical terms such as \"bandwidth,\" \"scripts,\" and \"data transfer,\" which may not be comprehensible by five-year-olds.\n"\
                    "2. Score: 2 out of 5. While the answer provides some information about the potential impact of net neutrality on websites like Reddit, it fails to use language that is appropriate for a five-year-old audience.\n"\
                    "3. Evidence:\n"\
                    "- \"Pictures and videos will be higher resolution.\"\n" \
                    "- \"Reddit itself will probably not change much, as it's an aggregation of content from elsewhere.\"\n" \
                    "- \"ISPs will be prompted to improve their service.\"\n"\
                    "- \"increase the service *for those who pay for preferential treatment*.\"\n" \
                    "- \"imgur and similar hosting sites will need to pony up the cash if they don't want to be left in the (relatively) dark ages of data transfer.\""

df1 = pd.DataFrame(
    data={
        'evaluation': evaluation_text1.split('\n'),
        "APPROVE ✅": [1, "", 1, 1, "", 1, "", 1],
        "DELETE ❎": ["", "", "", "", "untypical", "", "untypical", ""],
        "REVISE 🔀 or ADD 🆕": ["", "score:0", "", "", "", "", "", ""],
        "New Evaluation": ["",
                           "2. Score: 0. The answer fails to use language that is simple and appropriate for a five-year-old audience.",
                           "",
                           "", "", "", "", ""]
    }
)


eval_examples = [["The task is to provide an answer to a Reddit forum question that is comprehensible by five-year-olds.",
                   "reddit question",
                  "What would happen to a website like Reddit(a site that uses content from other websites) if Net Neutrality goes away?",
                  "answer",
                  "If Net Neutrality goes away, websites like Reddit would have to find a new way to share data with each other.",
                  ["2. Use examples or analogies that are relatable to a five-year-old’s experience."],
                  df,
                  "0"
                ],
                 [  "The task is to provide an answer to a Reddit forum question that is comprehensible by five-year-olds.",
                    "reddit question",
                    "What would happen to a website like Reddit(a site that uses content from other websites) if Net Neutrality goes away?",
                    "answer",
                    "As time goes on, the demands of the internet will increase.  Pictures and videos will be higher resolution.  Sites will have a greater number of scripts.  The whole experience will be \"richer\".  Reddit itself will probably not change much, as it's an aggregation of content from elsewhere, but all those sites that are linked are going to need more bandwidth. In order to cope with the increased demands, ISPs will be prompted to improve their service.  However... they don\'t NEED to.  Their audience needs their product and there isn\'t another option outside the big boys.  So what they\'ll likely do is increase the service *for those who pay for preferential treatment*.  That means links to big sites like the news outlets will keep up with the demands of the internet, while imgur and similar hosting sites will need to pony up the cash if they don\'t want to be left in the (relatively) dark ages of data transfer.",
                    ["1. Use simple and easy-to-understand language."],
                    df1,
                    "0"
                 ]]


df1= pd.DataFrame(data={
       "id": [1, 2, 3, 4, 5],
       "criteria":
           [
               "1. The answers should use simple and clear language that is easy for a five-year-old to understand.",
               "2. The answers should be concise and to-the-point, avoiding unnecessary details or complexities.",
               "3. The answers should be relevant to the question being asked, addressing the specific concerns or curiosities of the child.",
               "4. The answers should be accurate and factually correct, avoiding any misconceptions or inaccuracies that could confuse or misinform the child.",
               "5. The answers should be engaging and interactive, encouraging the child to ask further questions or explore the topic further."],
       "correction": ["approve", "approve", "approve", "approve", "approve"],  # pass, delete | xxx, revise | xxx, add | xxx  Note: xxx means reasons why delete, revise or add
       "new criteria": [
            "1. The answers should use simple and clear language that is easy for a five-year-old to understand.",
            "2. The answers should be concise and to-the-point, avoiding unnecessary details or complexities.",
            "3. The answers should be relevant to the question being asked, addressing the specific concerns or curiosities of the child.",
            "4. The answers should be accurate and factually correct, avoiding any misconceptions or inaccuracies that could confuse or misinform the child.",
            "5. The answers should be engaging and interactive, encouraging the child to ask further questions or explore the topic further."
    ]
   }, columns=["id", "criteria", "correction", "new criteria"])
df2 = pd.DataFrame(data={
             "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "criteria":
                 [
                     "1. The stories should be coherent and make sense in terms of the given beginning.",
                     "2. The stories should convey a clear message or lesson related to the daily event being described.",
                     "3. The stories should be engaging and hold the reader's attention.",
                     "4. The stories should be relatable and connect with the reader's personal experiences.",
                     "5. The stories should use descriptive language and vivid imagery to paint a clear picture in the reader's mind.",
                     "6. The stories should have a logical structure and flow smoothly from beginning to end.",
                     "7. The stories should be appropriate in tone and language for the intended audience.",
                     "8. The stories should demonstrate an understanding of human behavior and emotions related to the daily event being described.",
                     "9. The stories should be original and not rely on cliches or stereotypes.",
                     "10. The stories should be concise and to the point while still conveying the necessary information."],
             "correction": ["approve", "delete | lesson is not necessary, repeat with 8th one ", "approve", "delete | don't need to connect with reader", "delete | paint a clear picture in the reader's mind is not a must", "delete | same with the 3rd one", "approve", "approve", "revise | originality is not the must requirement", "approve"],
             "new criteria": [
                 "",
                 "",
                 "",
                 "",
                 "",
                 "",
                 "",
                 "",
                 "9. The stories should not rely on cliches or stereotypes.",
                 ""
             ]
},
columns=["id", "criteria", "correction", "new criteria"])
cap_examples = [
    ["The task is to provide an answer to a Reddit forum question that is comprehensible by five-year-olds.", "reddit question","answer", df1],
    ["The task is to provide a story for a given beginning which captures daily events.","beginning","story", df2]]

get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


# https://en.wikipedia.org/wiki/Markdown
# - This demo server. [[beta]](http://0.0.0.0:7860)
notice_markdown = ("""
# 💻🔧👷  Exploring the Reliability of Large Language Models as Evaluators
""")
# Human Evaluation with the Help of LLMs.
# ### Terms of use
# By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
# ### Choose a model to help you evaluate with

# - GPT-3: a 175-B language model
# - GPT-3.5: GPT-3 trained towards human preference by RLHF
# - GPT-3.5 turbo: a powerful conversational assistant

criteria_query_title = (
    """
    ## Step 1. List criteria needed by an NLP task.
    Type something under "APPROVE ✅" if a criterion is appropriate.
    Type **reasons** under "DELETE ❎" / "REVISE 🔀 or ADD 🆕" to shortly explain why a criterion is unnecessary / unreasonable or needed to add.
    """
)

criteria_evaluation_title = (
    """
    ## Step 2. Evaluate single task instance based on each criterion decided in Step 1.
    """
)

criteria_evaluation_title_t2 = (
    """
    ## Evaluate single task instance based on each criterion.
    Please CAREFULLY read the task instance and criterion before judging ChatGPT's evaluation.
    Please find ChatGPT's ERRORS as much as possible.
    """
)
#    **Evaluation Steps**:
#     1. Evaluate whether this answer satisfy the premise. Give a conclusion.
#     2. Assign a score for this answer on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the premise.
#     3. List evidence by quoting sentences of the answer to support your conclusion.

learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI. Please [contact us](mailto:qtleo@outlook) if you find any potential violation.
""")

sep_line = (
    """
    -----------
    """
)


code_highlight_css = (
"""
#chatbot .hll { background-color: #ffffcc }
#chatbot .c { color: #408080; font-style: italic }
#chatbot .err { border: 1px solid #FF0000 }
#chatbot .k { color: #008000; font-weight: bold }
#chatbot .o { color: #666666 }
#chatbot .ch { color: #408080; font-style: italic }
#chatbot .cm { color: #408080; font-style: italic }
#chatbot .cp { color: #BC7A00 }
#chatbot .cpf { color: #408080; font-style: italic }
#chatbot .c1 { color: #408080; font-style: italic }
#chatbot .cs { color: #408080; font-style: italic }
#chatbot .gd { color: #A00000 }
#chatbot .ge { font-style: italic }
#chatbot .gr { color: #FF0000 }
#chatbot .gh { color: #000080; font-weight: bold }
#chatbot .gi { color: #00A000 }
#chatbot .go { color: #888888 }
#chatbot .gp { color: #000080; font-weight: bold }
#chatbot .gs { font-weight: bold }
#chatbot .gu { color: #800080; font-weight: bold }
#chatbot .gt { color: #0044DD }
#chatbot .kc { color: #008000; font-weight: bold }
#chatbot .kd { color: #008000; font-weight: bold }
#chatbot .kn { color: #008000; font-weight: bold }
#chatbot .kp { color: #008000 }
#chatbot .kr { color: #008000; font-weight: bold }
#chatbot .kt { color: #B00040 }
#chatbot .m { color: #666666 }
#chatbot .s { color: #BA2121 }
#chatbot .na { color: #7D9029 }
#chatbot .nb { color: #008000 }
#chatbot .nc { color: #0000FF; font-weight: bold }
#chatbot .no { color: #880000 }
#chatbot .nd { color: #AA22FF }
#chatbot .ni { color: #999999; font-weight: bold }
#chatbot .ne { color: #D2413A; font-weight: bold }
#chatbot .nf { color: #0000FF }
#chatbot .nl { color: #A0A000 }
#chatbot .nn { color: #0000FF; font-weight: bold }
#chatbot .nt { color: #008000; font-weight: bold }
#chatbot .nv { color: #19177C }
#chatbot .ow { color: #AA22FF; font-weight: bold }
#chatbot .w { color: #bbbbbb }
#chatbot .mb { color: #666666 }
#chatbot .mf { color: #666666 }
#chatbot .mh { color: #666666 }
#chatbot .mi { color: #666666 }
#chatbot .mo { color: #666666 }
#chatbot .sa { color: #BA2121 }
#chatbot .sb { color: #BA2121 }
#chatbot .sc { color: #BA2121 }
#chatbot .dl { color: #BA2121 }
#chatbot .sd { color: #BA2121; font-style: italic }
#chatbot .s2 { color: #BA2121 }
#chatbot .se { color: #BB6622; font-weight: bold }
#chatbot .sh { color: #BA2121 }
#chatbot .si { color: #BB6688; font-weight: bold }
#chatbot .sx { color: #008000 }
#chatbot .sr { color: #BB6688 }
#chatbot .s1 { color: #BA2121 }
#chatbot .ss { color: #19177C }
#chatbot .bp { color: #008000 }
#chatbot .fm { color: #0000FF }
#chatbot .vc { color: #19177C }
#chatbot .vg { color: #19177C }
#chatbot .vi { color: #19177C }
#chatbot .vm { color: #19177C }
#chatbot .il { color: #666666 }
""")
#.highlight  { background: #f8f8f8; }



css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""
