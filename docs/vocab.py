import nltk, re, pprint
from nltk import word_tokenize
#nltk.download('punkt')
f = open('WordsPunctList.txt')
raw = f.read()
tokens = word_tokenize(raw)
#type(tokens)
words = [w.lower() for w in tokens]
#type(words)
#vocab = sorted(set(words))
#print(words)
coded = [item.encode('utf-8') for item in words]
#print(coded)
#OLD0 = []

'''
OLD1 = []
OLD0 = []
for word in words:
    L = 0
    for letter in word:
        L += 1
        #OLD0.append(letter)
        if L == len(word):
            OLD1.append(L)
#print(len(OLD0))
#print(len(OLD1))

#OLD1 is the int length of the words
#OLD0 is all the letters of the words, When done correctly it is the encoded versions
'''
