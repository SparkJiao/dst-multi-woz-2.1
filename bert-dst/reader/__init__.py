reader_dict = {

}


def initialize_reader(name, *args, **kwargs):
    return reader_dict[name](*args, **kwargs)
