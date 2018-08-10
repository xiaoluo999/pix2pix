import os


def assert_file_exist(rfile):
    if not os.path.exists(rfile):
        raise ValueError('file path not exists: ', rfile)


def list_to_file_line(rlist, rfile):
    with open(rfile, mode='w', encoding='utf8') as f:
        for element in rlist:
            f.write(element + '\n')


def file_line_to_list(rfile):
    assert_file_exist(rfile)
    with open(rfile, encoding='utf8') as f:
        return f.read().strip().split('\n')


def add_string_to_file_line(rstr, rfile):
    with open(rfile, mode='a', encoding='utf8') as f:
        f.write(rstr + '\n')


def file_line_to_key_value_list(rfile, line_spliter='\t'):
    assert_file_exist(rfile)
    line_list = file_line_to_list(rfile)
    key_value_list = []
    for line in line_list:
        key, value = line.split(line_spliter)
        key_value_list.append([key, value])
    return key_value_list
