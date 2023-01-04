import spacy
strr = "random string of values that needs to be tokenized"
nlp = spacy.load("en_core_web_sm")

vocab = {"UNK": 0, "string": 1, "that": 2, "to": 3, "needs": 4}
tokenized_string = [vocab.get(t.text.lower(), vocab["UNK"]) for t in nlp.tokenizer(strr)]
print(vocab)
dicct_list = [0,0,0,1,1,3]
aaa = [dicct_list.get() for id in dicct_list]