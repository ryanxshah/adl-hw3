from .cot import CoTModel
from .data import Dataset
import math
import json
from tqdm import tqdm

# black box this for now
def is_correct(pred: float, target: float, tol: float = 1e-2) -> bool:
    return math.isfinite(pred) and abs(pred - target) <= tol

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    trainset = Dataset("debug")
    model = CoTModel()

    output = []

    questions = [trainset[i][0] for i in range(len(trainset))]
    targets = [trainset[i][1] for i in range(len(trainset))]
    prompts = [model.format_prompt(question) for question in questions]

    generations = model.batched_generate(
        prompts,
        num_return_sequences=oversample,
        temperature=temperature
    )

    for question, target, q_generations in zip(questions, targets, generations):
        for generation in q_generations:
            if is_correct(model.parse_answer(generation), target):
                output.append([question, target, generation])
                break

    # check this later
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
        

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
