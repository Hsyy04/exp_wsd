from nltk.wsd import lesk
import nltk
# nltk.download('wordnet')
sent = 'I went to the bank to deposit my money'
answer = lesk(sent, "bank")
print(answer)