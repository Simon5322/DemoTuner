import re


def get_relevant_sections(doc_path="../doc/text.txt"):
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
            #print(str(key)+str(section))
    return relevant_sections

if __name__ == '__main__':
    ss = get_relevant_sections()
    for s in ss:
        print(s)


