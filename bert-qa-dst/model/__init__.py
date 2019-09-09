from .bert_qa_dst import BertDialogMatching

model_dict = {
    'dialog_matching': BertDialogMatching
}


def from_params(_config):
    _model_params = _config.pop("model")
    _model_name = _model_params.pop("name")
    return model_dict[_model_name].from_parms(_model_params)
