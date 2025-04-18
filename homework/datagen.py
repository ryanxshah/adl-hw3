

import json
import random
from tqdm import tqdm
from .data import Dataset  # or however your questions are loaded
from .cot import CoTModel
from .base_llm import BaseLLM

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
    model = BaseLLM()

    questions = Dataset("train")
    dataset = []

    for question, gt_answer in tqdm(questions):
        prompts = [model.format_prompt(question)] * num_return_sequences

        generations_nested = model.batched_generate(
            prompts,
            num_return_sequences=1,
            temperature=0.7,
        )
        generations = [g[0] if isinstance(g, list) else g for g in generations_nested]

        correct = []
        for gen in generations:
            pred = extract_answer(gen)
            if pred is not None and abs(pred - gt_answer) < 1e-3:
                correct.append(gen)

        if correct:
            reasoning = random.choice(correct)
            dataset.append([question, gt_answer, reasoning])

    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


#def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
