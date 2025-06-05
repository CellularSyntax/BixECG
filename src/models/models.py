from .bixecg import BiXLSTM, BiXLSTM2, FXLSTM, xLSTM, BimLSTM, BiXLSTM2_nonshared
from .jimenez_cnn import JimenezCNN1D
from .peimankar_cnn_bilstm import PeimankarCnnBilstm
from .liu_cnn_bilstm import LiuCNNBilstm
from .BiXLSTMClassifierFusion import BiXLSTMClassifierFusion

def get_model(name: str, **kwargs):
    """
    Return an instance of a model by name.

    Parameters:
        name (str): Model name (case-insensitive). One of:
            - 'bixlstm'
            - 'xlstm'
            - 'lightecgnet'
            - 'jimenezcnn1d'
        kwargs: Keyword arguments passed to the model constructor.

    Returns:
        torch.nn.Module: Instantiated model.

    Raises:
        ValueError: If an unknown model name is provided.
    """
    
    ####### getattr(module_model, model_class_name)(**kwargs)

    name = name.lower()
    model_registry = {
        "bixlstm": BiXLSTM,
        "bixlstmnew": BiXLSTM2,
        "bixlstmnew_nonshared": BiXLSTM2_nonshared,
        "bimlstm": BimLSTM,
        "fxlstm": FXLSTM,
        "xlstm": xLSTM,
        "liuecnnbilstm": LiuCNNBilstm,
        "jimenezcnn1d": JimenezCNN1D,
        "peimankarcnnbilstm": PeimankarCnnBilstm,
        "bixlstmfusion": BiXLSTMClassifierFusion
    }

    if name not in model_registry:
        raise ValueError(f"Unknown model '{name}'. Available models: {list(model_registry.keys())}")

    return model_registry[name](**kwargs)
