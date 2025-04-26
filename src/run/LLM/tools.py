import re
unit_table = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024,
        }

def get_conf_description(dbms_name, conf, current_values):
    description_dict = {}
    #description_str = ''
    dbms_name_List = ['mysql', 'pg']
    if dbms_name not in dbms_name_List:
        raise  KeyError(f'{dbms_name} is not valid dbms name')
    if dbms_name == 'pg':
        description_str = str({k: v['default'] for k, v in conf.items()})
        return description_str
    else:
        for param, v in conf.items():
            full_unit = conf.get(param)['unit']
            if full_unit != 'None':
                pattern = r'(\d+)(\w+)'
                result = re.search(pattern, full_unit)
                num = result.group(1)
                unit = result.group(2)  # KB or MB or GB
                description_dict[param] = str(int(current_values[param]) * int(num))+unit
            else:
                description_dict[param] = str(current_values[param])
    return str(description_dict)



def divide_united_value(united_value):
    pattern = r'(\d+)(\w+)'
    result = re.search(pattern, united_value)
    num = result.group(1)
    unit = result.group(2)  # KB or MB or GB
    return num, unit

def get_united_value(conf, param):
    full_unit = conf.get(param)['unit']
    if full_unit != 'None':
        pattern = r'(\d+)(\w+)'
        result = re.search(pattern, full_unit)
        num = result.group(1)
        unit = result.group(2)
        return int(num), unit
    else:
        return None
