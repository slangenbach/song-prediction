from pathlib import Path


# define constants
ROOT_PATH=Path.cwd().parent
RAW_DATA_PATH=ROOT_PATH/"data/raw"
INTERIM_DATA_PATH=ROOT_PATH/"data/interim"
MODEL_PATH=ROOT_PATH/"models"
    
def calc_layer_size(to, alpha: int = 5) -> list:
    """
    https://forums.fast.ai/t/an-attempt-to-find-the-right-hidden-layer-size-for-your-tabular-learner/45714
    """
    input_neurons = len(to.train.x_names)
    output_neurons = to.train.y.nunique()
    samples = len(to.train)
    return samples / (alpha * (input_neurons + output_neurons))