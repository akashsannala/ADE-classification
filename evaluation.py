test_text = "The patient was given 500mg of Paracetamol twice a day."

inputs = tokenizer(test_text, return_tensors="pt")
outputs = model(**inputs).logits
predictions = outputs.argmax(dim=-1)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = predictions[0].numpy()

for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")
