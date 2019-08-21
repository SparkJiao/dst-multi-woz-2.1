from .multi_woz import MultiWOZSpanReader

reader_dict = {
    "multi_woz": MultiWOZSpanReader
}


def initialize_reader(name, *args, **kwargs):
    return reader_dict[name](*args, **kwargs)
