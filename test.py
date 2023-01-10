from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import BLEUScore
from pprint import pprint

pred = 'three white dogs are sprinting after a man.'
target = 'two white cats are running after a man.'
target2 = 'two red dogs are running after a madiownmadiowadiownaiodwn wdioandiowanoi dwaiondoiwan odiwnaiodnw oindwainon.'
target3 = 'two black dogs'
target4 = 'three white dogs was sprinting after a man.'
target_list = [target, target2, target3, target4]

rogue = ROUGEScore()
bleu = BLEUScore(n_gram= 2)
r_scores = rogue(pred, target_list)
b_scores = bleu([pred], [target_list])


pprint(r_scores)
print(10*"--")
print(b_scores)


aaaaa = ["this", "is", "a", "list", "of", "tokens"]
ret_str = " ".join(aaaaa)
print(ret_str)

itarget = "this is a longer message which is the target"
ipred = "my prediction is short"

print(f"{'Prediction:':15} {ipred}")
print(f"{'target:':15} {itarget}")

import torchvision
vgg = torchvision.models.vgg19()  # load in encoder
print(vgg.named_parameters)