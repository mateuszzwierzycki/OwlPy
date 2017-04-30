import os
from enum import Enum

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()

    values = ""
    counter = 0
    length = len(local_device_protos)

    for device in local_device_protos:
        if device.device_type == "GPU":
            description = "Found " + device.physical_device_desc
            values += description
            if counter < length - 1: values += "\n"
        counter += 1

    # return [x.name for x in local_device_protos if x.device_type == 'GPU']
    return values


# this is taken from the C++ header file
# const int INFO = 0;            // base_logging::INFO;
# const int WARNING = 1;         // base_logging::WARNING;
# const int ERROR = 2;           // base_logging::ERROR;
# const int FATAL = 3;           // base_logging::FATAL;
# const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;


class MessageLevel(Enum):
    INFO = "0"
    WARNING = "1"
    ERROR = "2"
    FATAL = "3"
    NUM_SEVERITIES = "4"


def set_tf_message_level(level=MessageLevel.INFO):
    """Sets the minimal message level to be shown in the console.
      Args:
        level: MessageLevel enum.
      """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = level.value  # don't show the warnings in the console, only errors.
