from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import BLEUScore
from pprint import pprint

target = 'two white cats are running after a man.'
target2 = 'two red dogs are running after a madiownmadiowadiownaiodwn wdioandiowanoi dwaiondoiwan odiwnaiodnw oindwainon.'
target3 = 'two black dogs'
pred = 'three white dogs are sprinting after a man.'


rogue = ROUGEScore()
bleu = BLEUScore(n_gram= 4)
r_scores = rogue(pred, target)
b_scores = bleu(['three white dogs are sprinting after a man.'], [['two black dogs are running after a man', 'two red dogs are running after a man']])

metric = BLEUScore()
print(metric([pred], [[target, target2]]))


#print(r_scores)
print(10*"--")