from datasets import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True)
    
    labels = []
    for i, text in enumerate(examples["text"]):
        label_ids = [0] * len(tokenized_inputs["input_ids"][i])
        if examples["drug"][i] in text:
            start_idx = text.find(examples["drug"][i])
            end_idx = start_idx + len(examples["drug"][i])
            label_ids[start_idx:end_idx] = [1] * (end_idx - start_idx)  # Drug label

        if examples["dosage"][i] in text:
            start_idx = text.find(examples["dosage"][i])
            end_idx = start_idx + len(examples["dosage"][i])
            label_ids[start_idx:end_idx] = [2] * (end_idx - start_idx)  # Dosage label

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
