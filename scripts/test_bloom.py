# Test bloom modele
from transformers import AutoModelForSequenceTagging, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model = AutoModelForSequenceTagging.from_pretrained(
    "bigscience/bloom-560m",
    device_map="auto",
    torch_dtype="auto"
    )

def tokenize_function(instance):
    return tokenizer(instance, padding="max_length", truncation=True)

tokens = tokenize_function("This is an unbelievable test. Maintenant aussi incroyablement en français.")

print(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
