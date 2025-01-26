import pandas as pd
import argparse
import time
from openai import OpenAI
from tqdm import tqdm
import os

# Read data using
SYS_MSG = "Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"
eval_model = "gpt-3.5-turbo-0125"
temperature = 0
api_key = os.getenv("OPENAI_API_KEY")
folder_path = "Paper Experiment Results/New_filtered/IF"


def extract_final_response(response, prompt):
    return response[len(prompt) :]


def match_prompt_column(model):
    if "Llama-2" in model:
        return "llama2_chat_initial_prompt"
    elif "Mixtral" in model or "Mistral" in model:
        return "mixtral_instruct_initial_prompt"
    elif "Llama-3" in model:
        return "llama3_chat_initial_prompt"
    else:
        print("The model name didn't match anything, please check!!!!")
        return None


def extract_pure_response(row):
    model = row["model"]
    prompt_column = match_prompt_column(model)
    response = row["generated_response"]
    prompt = row[prompt_column]
    return extract_final_response(response=response, prompt=prompt)


def process_string(input_string):
    processed_string = input_string.strip("[]").replace("\\", "")
    questions = processed_string.split("\n")
    questions = [q.strip("'") for q in questions]
    return questions


def main():
    parser = argparse.ArgumentParser(description="Process file name for evaluation.")
    print("Starting...")
    parser.add_argument("file_name", type=str, help="File name to be processed")
    args = parser.parse_args()
    file_name = args.file_name
    output_file_name = file_name.replace(".csv", "_evaluated.csv")
    input_path = os.path.join(folder_path, file_name)
    output_path = os.path.join(folder_path, output_file_name)
    _data = pd.read_csv(input_path)
    _data["pure_response"] = _data.apply(extract_pure_response, axis=1)
    client = OpenAI(api_key=api_key)
    pbar = tqdm(total=len(_data))
    for index, entry in tqdm(_data.iterrows()):
        updated_row = entry.copy()
        input_task = entry["input"]
        output = entry["pure_response"]
        if output is None:  # skip if result hasn't been generated
            continue
        message = []
        answer = ""
        for question in process_string(entry["decomposed_questions"]):
            if len(message) == 0:
                if input_task:
                    content = f'{SYS_MSG}\n\nInput:\n"{input_task}"\n\nGenerated Text:\n"{output}"\n\nQuestion:\n{question}\n'
                else:
                    content = f'{SYS_MSG}\n\nGenerated Text:\n"{output}"\n\nQuestion:\n{question}\n'
            else:
                content = f"{question}\n"
            message.append({"role": "user", "content": content})
            # create a chat completion
            success = False
            early_stop = False
            while not success:
                try:
                    completion = client.chat.completions.create(
                        model=eval_model,
                        messages=message,
                        temperature=temperature,
                    )
                    generation = completion.choices[0].message.content
                    message.append({"role": "assistant", "content": generation})
                    # check if generation is yes or no
                    if generation.lower().startswith(
                        "yes"
                    ) or generation.lower().startswith("no"):
                        if generation.lower().startswith("yes"):
                            answer += "Yes\n"
                        else:
                            answer += "No\n"
                    else:
                        if "YES" in generation and "NO" not in generation:
                            answer += "Yes\n"
                        elif "YES" not in generation and "NO" in generation:
                            answer += "No\n"
                        else:
                            for msg in message:
                                print(msg["content"])
                            print("NO YES or NO answer!" + generation)
                            answer += "None\n"
                            early_stop = True
                            break
                    success = True
                except Exception as e:
                    print("ERROR!")
                    print(e)
                    print("Retry!")
                    time.sleep(20)

            # when no answer occurs, break the loop and continue to next instance
            if early_stop:
                break
        answer = answer[:-1]
        bool_results = []
        for i in answer.split("\n"):
            if i == "Yes":
                bool_results.append(True)
            elif i == "No":
                bool_results.append(False)
            else:
                bool_results.append(None)

        updated_row["eval"] = bool_results
        updated_row["messages_openai"] = message
        updated_dataframe = pd.DataFrame([updated_row])
        pbar.update(1)
        if not os.path.exists(output_path):
            updated_dataframe.to_csv(output_path, index=False, mode="w", header=True)
        else:
            print("append")
            updated_dataframe.to_csv(output_path, index=False, mode="a", header=False)


if __name__ == "__main__":
    main()
