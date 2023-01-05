import spacy
strr = "random string of values that needs to be tokenized"
nlp = spacy.load("en_core_web_sm")

vocab = {"UNK": 0, "string": 1, "that": 2, "to": 3, "needs": 4}
tokenized_string = [vocab.get(t.text.lower(), vocab["UNK"]) for t in nlp.tokenizer(strr)]

import torch
aa = torch.tensor([1,4,56,8,3,7], dtype= torch.int32)
for a in aa:
    print(a.item())