from .multi_woz_loader import MultiWOZReader


reader_dict = {
    'multi_woz': MultiWOZReader
}


def from_params(_config):
    reader_params = _config.pop("reader")
    reader_name = reader_params.pop("name")
    return reader_dict[reader_name].from_params(reader_params)

