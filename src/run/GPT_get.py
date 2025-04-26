import os
import re

import openai
from transformers import pipeline
#
from hintsClasses import conditionType
import tuningType

from hintsClasses import Condition

# 设置 OpenAI 认证密钥


conditionType = conditionType.conditionType
tuningType = tuningType.tuningType

# 定义输入文本
proPrompt = """
        我想要从文本中获得调优建议的多个三元组数据结构，即（配置参数的名称，调优的值，和调优的condition)，最后只需要给出python中的三元组数据结构，不要有其他多余的内容。推荐值可以是 
        1 一个绝对值，比如2gb，或一个范围[20，25]等	2 一个相对值，相对于别的参数，或者某一个指标，
        比如25%RAM等	3 调优趋势，increase，decrease。 调优的condition是在什么环境下，或给出了具体调优的数值。举个例子就是('join_buffer_size', 'increase', "read-heavy workload with tables that don't have indexes")
        如果一个参数有多条调优建议就生成多个三元组。文本是：
        """

text_generated = """
  [
    ('join_buffer_size', 'increase', "read-heavy workload with tables that don't have indexes"),
    ('join_buffer_size', 'decrease', "setting the value too high can cause significant performance drops"),
    ('sort_buffer_size', 'increase', "read-heavy workload (tables don't have indexes) and 'sort_merge_passes' increases quickly"),
    ('sort_buffer_size', 'decrease', "setting the value too high will affect performance for a smaller workload"),
    ('preload_buffer_size', 'modify', "workload is using indexes, as does an average WordPress workload")
  ]
  """

text = "[('work_mem', 'increase', 'for systems doing a lot of complex sorts, increasing sort memory can optimize configuration')]"
text1 = "[('percentage of created_tmp_disk_tables / (created_tmp_disk_tables + created_tmp_tables',), ('tmp_table_size', 'modify', 'if the percentage is 25% or greater')]"

conditionType = [item for sublist in conditionType for item in sublist]



# 获得gpt给的三元组(windows实现 已经保存在gpt_text下)
def send_gpt_request(prompt):
    openai.api_key = "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"
    openai.proxy = "http://127.0.0.1:10810"

    # prompt = proPrompt + text
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message


# 将文本的三元组转为数组(直接将多个转为数组)
def trans_plainText_to_dataStructure(text_data):
    params_pattern = r"\(([^)]+)\)"
    matches = re.findall(params_pattern, text_data)
    data = [tuple(match.strip().strip("'\"") for match in match_group.split(",")) for match_group in
            matches]  # text_generated 转为数组结构
    return data

def gpt_text_to_excel():
    text_path = '../doc/Postgres/gpt_text'



"""
将数组转为具体标签
conditionType: index,read heavy 
data: data structure of plain text
condition_num_limit: 考虑多少个condition
'join_buffer_size': [{'value': 'increase', 'condition': ['heavy read']}, {'value': 'decrease', 'condition': []}], 'sort_buffer_size': [{'value': 'increase', 'condition': ['heavy workload']}
"""


def trans_plainData_to_formal(conditionType, data, condition_num_limit: 1):
    if data is None:
        return
    zsc_pipeline = pipeline(
        'zero-shot-classification',
        model="facebook/bart-large-mnli")
    # condition_num_limit = 1  # 1只考虑一个
    hints = {}
    # print(conditionType)
    for sourcehint in data:
        if len(sourcehint)!= 3:
            continue
        if sourcehint[0] == '' or sourcehint[1] == '':
            continue
        if not sourcehint[0] in hints:
            hints[sourcehint[0]] = []
        # hint = {"conf": sourcehint[0]}
        hint = {}
        # get tuning value , like increase
        tuning_value_result = zsc_pipeline(sourcehint[1], tuningType)
        if tuning_value_result["scores"][0] > 0.4:
            hint["value"] = tuning_value_result["labels"][0]

        # get tuning condition
        condition = []
        if sourcehint[2] != '':
            condition_result = zsc_pipeline(sourcehint[2], conditionType)
            # print(condition_result)

            for i in range(condition_num_limit):
                if condition_result["scores"][i] > 0.1:
                    condition.append(condition_result["labels"][i])
        else:
            condition.append("NO_CONDITION")
        hint["condition"] = condition

        # print(hint)
        hints[sourcehint[0]].append(hint)
        # print(hints)
    return hints

    # print(str(result["sequence"])+"  "+str(result["labels"][0])+"  "+ str(result["scores"][0])+str(result["labels"][1])+"  "+ str(result["scores"][1]))


# 提取生成的文本结果
# result = response.choices[0].text.strip()

# #
# 从数据结构中获得所有参数名
def get_parameters_in_text(data, should_verify_parameter):
    all_params = []
    seen = set()
    for tup in data:

        param = tup[0]
        # print(param)
        if param not in seen:
            if should_verify_parameter:
                if verifyParam(param):
                    all_params.append(param)
                    seen.add(param)
                # else:
                #     return all_params
            else:
                all_params.append(param)
                seen.add(param)
    return all_params


def verifyParam(param):
    return True

# 结合上面两个函数
def load_gptText_to_hints():
    gpt_path = "../doc/gpt_text/"
    file_names = os.listdir(gpt_path)
    tups = []
    hints = []
    # 将原始的文本tuple转为tuple并且放入tups中
    for file in file_names:
        file_path = os.path.join(gpt_path, file)
        with open(file_path, "r", errors='ignore') as f:
            lines = f.readlines()
            tups.extend(trans_plainText_to_dataStructure(str(lines)))  # 获得数组化的三元组
            # 将三元组转化为hint

    # print(tups)
    # print(type(tups[0]))
    # print(len(tups))
    batch_size = 10
    condition_limit = 1
    for i in range(0, len(tups), batch_size):
        batch_tups = tups[i:i + batch_size]
        print(batch_tups)
        batch_hints = trans_plainData_to_formal(conditionType, batch_tups, condition_limit)
        print(batch_hints)
        hints.extend(batch_hints)
    return hints


def get_hints():
    data = trans_plainText_to_dataStructure(text)
    formal_data = trans_plainData_to_formal(conditionType, data, 2)
    return formal_data





if __name__ == "__main__":
    hints = load_gptText_to_hints()

