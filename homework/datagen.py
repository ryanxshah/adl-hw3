from .cot import CoTModel
from .data import Dataset

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    trainset = Dataset("train")
    model = CoTModel()

    # a list of all the questions as strings
    questions = [trainset[i][0] for i in range(len(trainset))]

    # a list of the formatted prompts (1 per question)
    prompts = [model.format_prompt(question) for question in questions]

    # a list of lists - each sublist is all the generations for an individual question
    generations = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)









if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
