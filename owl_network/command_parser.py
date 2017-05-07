import sys


def parse_incoming(end_at_message):
    output_dict = {}

    while True:
        current_line = sys.stdin.readline()
        current_line = current_line.strip()
        if current_line != end_at_message:
            current_line = current_line.split(" ")
            print(current_line)
            output_dict[current_line[0]] = current_line[1]
        else:
            break

    print("Commands parsed")
    return output_dict


def try_get(dictionary, value_name, default_value):
    if value_name in dictionary:
        return dictionary[value_name]
    else:
        return default_value


def print_u(*args):
    print(*args, flush=True)
