import os
import re

import requests
from bs4 import BeautifulSoup

"""
save config tuning docs from web to doc/sourceWebTxt
"""


def save_text_from_web():
    # 指定目标网页的URL
    web_urls_path = "../doc/urls.txt"
    urls = []
    with open(web_urls_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        urls.append(line)
    if not urls:
        print("no urls")
    save_path = "../doc/sourceWebtxt/"
    # 发送HTTP请求以获取页面内容
    index = 0
    for url in urls:
        response = requests.get(url)

        # 检查请求是否成功
        if response.status_code == 200:
            # 使用BeautifulSoup解析HTML内容
            soup = BeautifulSoup(response.text, 'html.parser')
            # 查找网页上的文字内容
            text_content = soup.get_text()
            # with open("../doc/text.txt", "r") as f:
            #     text_content = f.read()
            text_content = re.sub(r'\n+', '\n', text_content).strip()
            match = re.search(r'//(.*?)/', url)
            file_name = str(match.group(1))
            # text_content = text_content.replace(" ", "")
            # 打印或处理文本内容
            index = index + 1
            doc_path = save_path + file_name + ".txt"
            with open(doc_path, "w") as f:
                f.write(text_content)
            print("doc save succese   " + url)
        else:
            print('无法访问网页  ' + url + str(response.status_code))


def get_relevant_sections(doc_path):
    with open(doc_path, "r") as f:
        text = f.read()
    relevant_sections = []
    param_reg = r'[a-z_]+_[a-z]+'

    sections = text.split('\n')
    key = 0
    for section in sections:
        key = key + 1
        if re.findall(param_reg, section):
            relevant_sections.append(section)
            # print(str(key)+str(section))
    return relevant_sections


def trans_all_source_to_relevant():
    source_path = "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/doc/sourceWebtxt"

    for root, dirs, files in os.walk(source_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            relevants = get_relevant_sections(file_path)

            relevant_path = "../doc/releavant_text/"
            if not os.path.exists(relevant_path):
                os.makedirs(relevant_path)
            for r in relevants:
                with open(relevant_path + file_name, "a") as f:
                    f.write(r + "\n")


if __name__ == '__main__':
    save_text_from_web()
    #trans_all_source_to_relevant()
