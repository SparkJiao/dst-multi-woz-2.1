from pytorch_transformers.modeling_bert import BertPreTrainedModel

model_dict = {

}


def initialize_model(name, *args, state_dict=None, **kwargs):
    return model_dict[name].from_pretrained(*args, state_dict=state_dict, **kwargs)
