# Test local trained model
from transformers import AutoModelForSequenceClassification, XLMRobertaForTokenClassification, RobertaForSequenceClassification, AutoTokenizer
from transformers import TokenClassificationPipeline

from jiant.proj.main.modeling.primary import JiantRobertaModel

import torch
from torch import tensor

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
# tokenizer.tokenize(str) -> subtokens
# tokenizer.encode(str) -> subtokens ids
# tokenizer.decode(subtokens ids) -> string

PATH = "./runs/run_disrpt23_eng_rst_gum_split_xlm-roberta-large/1684944862/"

# model = AutoModelForSequenceClassification.from_pretrained(path)
#model = RobertaForSequenceClassification.from_pretrained("../runs/run_disrpt23_eng_rst_gum_split_xlm-roberta-large/1684944862/")
model = XLMRobertaForTokenClassification.from_pretrained(PATH)
model.eval()

# will lack the config file -> should be the roberta original config (?untested)
model2 = torch.load(PATH+"/best_model.p")
model2.eval()

# does not include xlmroberta (yet?)
# pipeline = TokenClassificationPipeline(model=model,tokenizer=tokenizer,aggregation_strategy="simple")

#def tokenize_function(instance):
#    return tokenizer(instance, padding="max_length", truncation=True,return_tensors="pt")




document = "This is an unbelievable test. Now we will make another one"

# batch of 1
inputs = tokenizer(document, padding="max_length", truncation=True, return_tensors="pt")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print(inputs["input_ids"][0,:50])

labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs)
loss, logits = outputs[:2]
#tokens = pipeline(sentence)
#print(tokens)
