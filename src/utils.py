import logging

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False



def update_dict_from_args(dictionary, args):
    """
    Updates a dict object based on arguments on the format: -param1 VALUE -param2 VALUE ... -paramN VALUE

    :param dictionary The dict to update
    :param args the arguments as a list, retrieved directly from sys.argv
    """
    for i in range(len(args)):
        if args[i][0] == '-' and args[i][1] != '-':
            val = args[i + 1]
            if is_int(val):
                val = int(val)
            elif is_float(val):
                val = float(val)
            dictionary[args[i][1:]] = val


def get_logging_level(logging_str):
    """
    Returns a logging level object based on logging_str

    :param logging_str String indicating logging level. Example "warning"
    """
    logging_level = logging.DEBUG
    if logging_str == "info":
        logging_level = logging.INFO
    elif logging_str == "warning":
        logging_level = logging.WARNING
    elif logging_str == "error":
        logging_level = logging.ERROR
    elif logging_str == "critical":
        logging_level = logging.CRITICAL
    return logging_level


def is_empty(s):
    """
    Returns true if str is empty otherwise false. A string is defined as empty if it is None or only consists
    of whitespace.

    :param s: The str to test
    """
    return s is None or len(s.strip()) <= 0
