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

    trainset = Dataset("train")
    model = CoTModel()

    """
    # a list of all the questions as strings
    questions = [trainset[i][0] for i in range(len(trainset))]

    # a list of the formatted prompts (1 per question)
    prompts = [model.format_prompt(question) for question in questions]

    # a list of lists - each sublist is all the generations for an individual question
    generations = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)
    """

    output = []

    for question, target in tqdm(trainset):
        prompt = model.format_prompt(question)
        generations = model.batched_generate(
            prompt,
            num_return_sequences=oversample,
            temperature=temperature
        )

        for generation in generations:
            answer = model.parse_answer(generation)
            # need to round?
            if is_correct(answer, target):
                output.append([question, target, generation])
                break

    # check this later
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)









if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
