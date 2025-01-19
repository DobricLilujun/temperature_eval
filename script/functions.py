import numpy as np
import torch

from transformers import (
    BertTokenizer,
)
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def bertClassifier(
    input_text, model_path, tokenizer_path, max_padding=512
):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    model = torch.load(model_path)
    input_ids = []
    attention_masks = []
    texts = [input_text]
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_padding,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor([[0, 1, 2, 3, 4, 5]]).to(torch.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    b_input_ids = input_ids.to(device)
    b_input_mask = attention_masks.to(device)

    with torch.no_grad():
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = output.logits
    logits = logits.detach().cpu().numpy()

    probabilities = np.exp(
        logits - np.max(logits, axis=1, keepdims=True)
    )  # Stabilized softmax
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

    # Get the mapping of categories to codes before converting to codes
    ability_mapping = {0: "CR", 1: "CT", 2: "ICL", 3: "IF", 4: "MT", 5: "SUM"}

    output = ""
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        for code, category in ability_mapping.items():
            output += f"{category}: {prob[code]:.2f} "
            prob_dict[category] = prob[code]

    return output, prob_dict


def bertClassifierList(
    input_text_list, model_path, tokenizer_path, max_padding=512
):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    model = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    all_outputs = []
    all_prob_dicts = []

    for input_text in input_text_list:
        input_ids = []
        attention_masks = []

        # Encode the current text
        encoded_dict = tokenizer.encode_plus(
            input_text,  # Sentence to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_padding,  # Pad & truncate to max length
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attention masks
            return_tensors="pt",  # Return PyTorch tensors
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

        # Convert the lists into tensors
        input_ids = torch.cat(input_ids, dim=0).to(device)
        attention_masks = torch.cat(attention_masks, dim=0).to(device)
        # Perform forward pass without gradient calculation
        with torch.no_grad():
            output = model(
                input_ids, token_type_ids=None, attention_mask=attention_masks
            )

        logits = output.logits
        logits = logits.detach().cpu().numpy()

        # Stabilized softmax for probabilities
        probabilities = np.exp(
            logits - np.max(logits, axis=1, keepdims=True)
        )  # Stabilized softmax
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

        # Map probabilities to categories
        ability_mapping = {0: "CR", 1: "CT", 2: "ICL", 3: "IF", 4: "MT", 5: "SUM"}
        prob_dict = {
            ability_mapping[code]: prob for code, prob in enumerate(probabilities[0])
        }

        # Prepare output and probability dictionary
        output_text = "\n ".join(
            [f"{cat}: {prob:.2f}" for cat, prob in prob_dict.items()]
        )
        all_outputs.append(output_text)
        all_prob_dicts.append(prob_dict)
        agg_prob_dict = {}
        for _, ability in ability_mapping.items():
            agg_prob = 0
            for prob_dict in all_prob_dicts:
                agg_prob += prob_dict[ability]
            agg_prob_dict[ability] = agg_prob
        output_str = ""
        for key in agg_prob_dict:
            agg_prob_dict[key] /= len(input_text_list)
            sub_string = "{:.2f}".format(round(agg_prob_dict[key], 2))
            output_str += f"{key}: {sub_string} "

    return output, agg_prob_dict

