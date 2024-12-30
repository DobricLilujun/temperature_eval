import json
import os
import time
from tqdm import tqdm
import requests
import openai


class EVALUATOR:
    def __init__(self, server_url, model_name):
        """
        Initialize the evaluator.

        :param input_path: Path to the input JSONL file.
        :param output_path: Path to the output JSONL file.
        :param eval_model: Model name to use for evaluation (e.g., 'gpt-3.5-turbo').
        :param api_client: API client instance (OpenAI or vLLM).
        :param temperature: Temperature setting for the model (default is 0).
        """
        self.server_url = server_url
        # System message for evaluation task

        self.model_name = model_name
        self.options = {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "seed": 47,  # Default
            "n": 1,
        }
        self.openai_api_key = None
        self.evaluation_method = None

    def generate_text_with_vllm(self, prompt):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        payload.update(self.options)
        response = requests.post(self.server_url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    def evaluate(self, question_content):

        success = False
        max_attempts = 3
        times = 0
        while not success and times < max_attempts:
            try:
                # Use OpenAI or vLLM API for generating evaluation results
                if self.openai_api_key is not None:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question_content},
                    ]
                    completion = openai.ChatCompletion.create(
                        model=self.eval_model,
                        messages=messages,
                        temperature=self.temperature,
                    )
                    generation = completion.choices[0].message.content.strip()
                    times += 1

                # Use VLLM API for generating evaluation results
                else:
                    generation = self.generate_text_with_vllm(
                        self.model_name, question_content
                    )
                    times += 1
                label, generation = self.evaluation_method(generation)
                if label is not None:
                    success = True
                    return label, generation

            except Exception as e:
                print("ERROR!")
                print(e)
                print("Retrying...")
                time.sleep(1)
