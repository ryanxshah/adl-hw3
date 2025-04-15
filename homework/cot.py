from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {
                "role": "system",
                "content": "You're a helpful assistant that explains and answers unit conversions. Be concise, and give the final numeric answer inside <answer> tags."
            },
            {
                "role": "user",
                "content": "Convert 3 meters to feet."
            },
            {
                "role": "assistant",
                "content": "We know that 1 meter is approximately 3.281 feet. So 3 meters is 3 × 3.281 = 9.843 feet. <answer>9.843</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
