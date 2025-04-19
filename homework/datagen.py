from .cot import CoTModel
from .data import Dataset

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    trainset = Dataset("train")
    cot_model = CoTModel()

    prompts = ...




if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
