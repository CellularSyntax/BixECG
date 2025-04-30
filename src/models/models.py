from bixecglstm import BiXLSTM, xLSTM
from jimenez_cnn_1d import JimenezCNN1D

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
    name = name.lower()
    model_registry = {
        "bixlstm": BiXLSTM,
        "xlstm": xLSTM,
        "jimenezcnn1d": JimenezCNN1D,
    }

    if name not in model_registry:
        raise ValueError(f"Unknown model '{name}'. Available models: {list(model_registry.keys())}")

    return model_registry[name](**kwargs)
