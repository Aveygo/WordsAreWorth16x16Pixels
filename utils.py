import torch, numpy as np
import os

MODEL_NAMES = ["GPT", "PACING", "GPT_PRETRAINED"]

def latest_state_dict(model_name:str):
    """
    Find the latest state dict for a model
    """

    # Finding latest folder for model
    folders = os.listdir("sessions")
    folders = [f for f in folders if f.startswith(model_name)]
    folders = sorted(folders, key=lambda x: int(x.split("_")[-2])) # Sort by timestamp (modelname_timestamp_randomid)
    
    if len(folders) == 0:
        raise Exception("No folders found for model.")

    latest_folder = folders[-1]
    
    # Loading and renaming state dict
    state_dict_path = os.path.join("sessions", latest_folder, "model_state_dict.pt")
    state_dict = torch.load(state_dict_path)
    renamed = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    return renamed

def set_seed(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(model_name:str):
    if model_name == MODEL_NAMES[0]:
        from models.gpt_model import Generator
    elif model_name == MODEL_NAMES[1]:
        from models.pacing_model import Generator
    elif model_name == MODEL_NAMES[2]:
        from models.pretrained_gpt_wrapper import Generator
    else:
        raise Exception("Model not found.")
    
    return Generator()