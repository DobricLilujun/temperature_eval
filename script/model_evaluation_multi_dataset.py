import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import json
from utils.evaluators import EVALUATOR
from langchain import PromptTemplate
from tqdm import tqdm
import re
import string
from collections import Counter
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


LIST_DATASET_CSV = ["CR", "CT", "ICL", "IF", "MT", "SUMM"]  # Dataset list
file_path = "/home/llama/Personal_Directories/srb/temperature_eval/data/output/vllm_exp_dataset_csv_Llama-3.2-3B-Instruct-awq__20241226_023606.jsonl"
df = pd.read_json(file_path, lines=True)
evaluator_model_path = "/home/llama/models/base_models/Llama-3.3-70B-Instruct"

model_name = "Llama-3.2-3B-Instruct"
evaluator_model_name = "Llama-3.3-70B-Instruct"
server_url = "http://0.0.0.0:4361/v1/chat/completions"

model_name = os.path.splitext(os.path.basename(file_path))[0]
evaluator = EVALUATOR(
    server_url=server_url,
    model_name=evaluator_model_path,
)

with open(
    "/home/llama/Personal_Directories/srb/temperature_eval/data/ttcw_all_tests.json",
    "r",
) as f:
    tests = json.load(f)
output_folder = (
    "/home/llama/Personal_Directories/srb/temperature_eval/data/output/evaluation/"
)
output_prefix = f"evaluated_{evaluator_model_name}_{model_name}"


def evaluate_response_CR(generation):
    # check if generation is yes or no
    if generation.lower().startswith("yes") or generation.lower().startswith("no"):
        if generation.lower().startswith("yes"):
            return True, generation
        else:
            return False, generation
    else:
        if "yes" in generation.lower() and "no" not in generation.lower():
            return True, generation
        elif "yes" not in generation.lower() and "np" in generation.lower():
            return False, generation
        else:
            # print("NO YES or NO answer!" + generation)
            return None, generation


def evaluate_response_CT(generation):
    # check if generation is yes or no
    if generation.lower().startswith("yes") or generation.lower().startswith("no"):
        if generation.lower().startswith("yes"):
            return True, generation
        else:
            return False, generation
    else:
        if "yes" in generation.lower() and "no" not in generation.lower():
            return True, generation
        elif "yes" not in generation.lower() and "np" in generation.lower():
            return False, generation
        else:
            # print("NO YES or NO answer!" + generation)
            return None, generation


def evaluate_response_IF(generation):
    # check if generation is yes or no
    if generation.lower().startswith("yes") or generation.lower().startswith("no"):
        if generation.lower().startswith("yes"):
            return True, generation
        else:
            return False, generation
    else:
        if "yes" in generation.lower() and "no" not in generation.lower():
            return True, generation
        elif "yes" not in generation.lower() and "np" in generation.lower():
            return False, generation
        else:
            # print("NO YES or NO answer!" + generation)
            return None, generation


evaluate_prompt_CT = """You are given a creative short-story. Read it carefully. You are then given some background about specific aspects of creative writing, as well as a binary (Yes/No) question. Your objective is to use the background information to answer the question about the story. Start your answer with "Yes" or "No". You can optionally then provide a short explanation for your answer.

==========
Story:
{story}
==========
Background:
{background}

==========
Question: {question}

Remember to start your answer with Yes or No. You can optionally then provide a short explanation for your answer.
"""


def full_prompt2context(full_prompt):
    lines = full_prompt.strip().split("\n")
    kept1 = "\n".join(lines[:-1]).strip().split("\n")
    kept2 = kept1[:-1]
    return "\n".join(kept2).strip()


for test in tests:
    test["expanded_context"] = full_prompt2context(test["full_prompt"])

evaluate_prompt_template_CT = PromptTemplate.from_template(evaluate_prompt_CT)


def classification_score(prediction, ground_truth, all_classes):
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


categories = [
    "Food",
    "Date",
    "Order, rank",
    "Speed",
    "Disease and medicine",
    "Word with a special property",
    "Abbreviation",
    "Language",
    "Letter like a-z",
    "Other entity",
    "Animal",
    "Expression abbreviated",
    "Price",
    "Techniques and method",
    "Musical instrument",
    "Mountain",
    "Currency name",
    "Event",
    "Product",
    "State",
    "Individual",
    "Organ of body",
    "Reason",
    "Manner of an action",
    "City",
    "Religion",
    "Invention, book and other creative piece",
    "Distance, linear measure",
    "Temperature",
    "Postcode or other code",
    "Size, area and volume",
    "Sport",
    "Country",
    "Other location",
    "Lasting time of somethin",
    "Equivalent term",
    "Description of something",
    "Weight",
    "Vehicle",
    "Color",
    "Other number",
    "Definition of something",
    "Element and substance",
    "Description of a person",
    "Symbols and sign",
    "Number of something",
    "Plant",
    "Percent, fraction",
    "Group or organization of person",
    "Title of a person",
]


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def process_string(input_string):
    processed_string = input_string.strip("[]").replace("\\", "")
    questions = processed_string.split("\n")
    questions = [q.strip("'") for q in questions]
    return questions


scorer = BERTScorer(model_type="bert-large-uncased")
# bleu = BLEU(tokenize="flores101", effective_order=True)


def calculate_rouge_scores(generation, reference):
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    scores = scorer.score(generation, reference)
    precision = scores["rougeL"][0]
    recall = scores["rougeL"][1]
    fmeasure = scores["rougeL"][2]
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


from tqdm import tqdm


# def compute_bleu_score(row):
#     generated_response = row["generate_response"]
#     reference_answer = row["target"]
#     # Calculate BLEU score
#     score = bleu.sentence_score(
#         hypothesis=generated_response, references=[reference_answer]
#     ).score
#     return score


def compute_rouge_score(row):
    generated_response = row["generate_response"]
    reference_answer = row["target"]
    # Calculate ROUGE scores
    answer_dict = calculate_rouge_scores(generated_response, reference_answer)
    return answer_dict["fmeasure"]  # Return the f-measure score


def compute_icl_score(row):
    generated_response = row["generate_response"]
    reference_answer = row["target"].strip("[]' ")  # Clean up target string
    # Assume classification_score is a defined function
    score = classification_score(generated_response, reference_answer, categories)
    return score


# tqdm.pandas()
# MT_df = df[df["category"] == "MT"]
# MT_df["MT_accuracy"] = MT_df.progress_apply(compute_bleu_score, axis=1)
# MT_df.to_json(f"{output_folder}/{output_prefix}_MT.jsonl", lines=True, orient="records")

tqdm.pandas()
SUMM_df = df[df["category"] == "SUMM"]
SUMM_df["SUMM_accuracy"] = SUMM_df.progress_apply(compute_rouge_score, axis=1)
SUMM_df.to_json(
    f"{output_folder}/{output_prefix}_SUMM.jsonl", lines=True, orient="records"
)

tqdm.pandas()
ICL_df = df[df["category"] == "ICL"]
ICL_df["ICL_accuracy"] = ICL_df.progress_apply(compute_icl_score, axis=1)
ICL_df.to_json(
    f"{output_folder}/{output_prefix}_ICL.jsonl", lines=True, orient="records"
)


# CR Dataset Processing
for i, row in tqdm(
    df[df["category"] == "CR"].iterrows(),
    total=df[df["category"] == "CR"].shape[0],
    desc="Processing CR Rows",
    unit="Row",
):
    update_row = row.copy()
    # Get Basic Information
    SYS_MSG = """Evaluate the provided answer (if available) and the generated answer, and respond to the following question only with either 'Yes' or 'No'. Choose 'Yes' if both answers convey the same meaning. Choose 'No' if the meanings of the two answers differ."""
    generated_response = row["generate_response"]
    reference_answer = row["target"]
    message_content = (
        f'{SYS_MSG}\nAnswer 1:\n"{generated_response}"\nAnswer 2:\n{reference_answer}\n'
    )

    # Do the Evaluation
    evaluator.evaluation_method = evaluate_response_CR
    label, generated_response = evaluator.evaluate(question_content=message_content)
    update_row["CR_question_evaluation"] = message_content
    update_row["CR_label"] = label
    update_row["CR_evaluation_response"] = generated_response
    update_row["CR_accuracy"] = int(label) if label is not None else 0
    with open(f"{output_folder}/{output_prefix}_CR.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(update_row.to_dict(), ensure_ascii=False) + "\n")

# CT Dataset Processing
for i, row in tqdm(
    df[df["category"] == "CT"].iterrows(),
    total=df[df["category"] == "CT"].shape[0],
    desc="Processing CT Rows",
    unit="Row",
):
    update_row = row.copy()
    generated_response = row["generate_response"]

    evaluator.evaluation_method = evaluate_response_CT
    labels = []
    gen_eval_responses = []
    message_contents = []
    for question in tests:
        background = question["expanded_context"]
        Q = question["question"]
        message_content = evaluate_prompt_template_CT.format(
            story=generated_response,
            background=background,
            question=Q,
        )
        # evaluator.openai_api_key = api_key
        label, gen_eval_response = evaluator.evaluate(question_content=message_content)
        labels.append(label)
        gen_eval_responses.append(gen_eval_response)
        message_contents.append(message_content)

    update_row["CT_Label_{i}"] = str(labels)
    update_row["CT_Q{i}_responses"] = str(gen_eval_responses)
    update_row["CT_question_evaluation"] = str(message_contents)
    true_count = labels.count(True)
    accuracy = true_count / len(labels) if labels else 0
    update_row["CT_accuracy"] = accuracy
    with open(f"{output_folder}/{output_prefix}_CT.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(update_row.to_dict(), ensure_ascii=False) + "\n")
    # evaluator.openai_api_key = None

# IF Dataset Processing
for i, row in tqdm(
    df[df["category"] == "IF"].iterrows(),
    total=df[df["category"] == "IF"].shape[0],
    desc="Processing IF Rows",
    unit="Row",
):
    update_row = row.copy()
    SYS_MSG = "Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"
    generated_response = row["generate_response"]
    reference_answer = row["target"]
    prompt_input = row["input"]
    gen_eval_responses = []
    labels = []
    message_contents = []
    for question in process_string(reference_answer):

        content = f'{SYS_MSG}\n\nGenerated Text:\n"{generated_response}"\n\nQuestion:\n{question}\n'
        evaluator.evaluation_method = evaluate_response_IF
        label, gen_eval_response = evaluator.evaluate(question_content=content)
        labels.append(label)
        gen_eval_responses.append(gen_eval_response)
        message_contents.append(content)

    update_row["IF_question_evaluation"] = str(message_contents)
    true_count = labels.count(True)
    accuracy = true_count / len(labels) if labels else 0
    update_row["IF_accuracy"] = accuracy
    update_row["IF_label"] = str(labels)
    update_row["IF_evaluation_response"] = str(gen_eval_responses)
    with open(f"{output_folder}/{output_prefix}_IF.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(update_row.to_dict(), ensure_ascii=False) + "\n")
