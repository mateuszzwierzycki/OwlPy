import sys


def parse_incoming(end_at_message):
    output_dict = {}
    count = 0
    
    def is_int(value):
        try:
            int(value)
            return True
        except:
            return False
    
    def is_float(value):
        try:
            float(value)
            return True
        except:
            return False
    
    while True:
        current_line = sys.stdin.readline()
        current_line = current_line.strip()
        if current_line == '':
            # breaks if it reads an empty line OR finds the end_at_message
            break
        if current_line != end_at_message:
            current_line = current_line.split(" ")
            print(current_line)
            if is_int(current_line[1]) == True:
                output_dict[current_line[0]] = current_line[1]
            if is_float(current_line[1]) == True:
                output_dict[current_line[0]] = current_line[1]
            else:
                output_dict[current_line[0]] = current_line[1]
            count += 1
        elif count >= 20:
            # arbitrary limit of 20 max lines
            break
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
