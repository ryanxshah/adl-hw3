

import json
import random
from tqdm import tqdm
from .data import Dataset  # or however your questions are loaded
from .base_llm import CoTModel

def extract_answer(text: str) -> float | None:
    """Extract the float value from <answer>...</answer>."""
    import re
    match = re.search(r"<answer>(.*?)</answer>", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def generate_dataset(path: str = "data/rft.json", num_return_sequences: int = 10):
    model = CoTModel(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    questions = Dataset("train")
    dataset = []

    for question, gt_answer in tqdm(questions):
        generations = model.batched_generate(
            [question],
            num_return_sequences=num_return_sequences,
            temperature=0.7,
        )

        # extract generations and filter
        correct_samples = []
        for gen in generations:
            pred = extract_answer(gen)
            if pred is not None and abs(pred - gt_answer) < 1e-3:
                correct_samples.append(gen)

        if correct_samples:
            reasoning = random.choice(correct_samples)
            dataset.append([question, gt_answer, reasoning])

    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)

#def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
