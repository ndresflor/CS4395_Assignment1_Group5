from math import log, exp
from collections import Counter, defaultdict
from re import sub

# this function lowercases all the words and removes punctuation using regex
def preprocessing(fileName):
    with open(fileName, 'r') as file:
        lines = file.readlines()
        sentences = []

        for line in lines:
            sentence = line.strip()
            cleaned_sentence = sub(r'[^\w\s]', "", sentence.lower())
            words = cleaned_sentence.split()
            sentences.append(words)
    return sentences

data = preprocessing('train.txt')

# replace words with frequency lower than n with <UNK>
# Count frequency of words
word_counts = defaultdict(int)
for words in data:
  for w in words:
    word_counts[w] += 1

threshold = 5 # if frequency of word < threshold, replace with <UNK>

# remove words under threshold frequency
vocab = set() # will contain a set of known words
for key, value in word_counts.items():
  if value >= threshold:
    vocab.add(key)

# get data as unigrams
unigrams = defaultdict(int)
total_unigrams = 0

for words in data:
    for w in words:
        if w in vocab:
          unigrams[w] += 1
        else:
          unigrams['UNK'] += 1
        total_unigrams += 1

# calculate the probability of each unigram
for unigram in unigrams:
  unigrams[unigram] /= total_unigrams

# get data as bigrams
# For each word, there is a nested dict with counts for all possible second words
bigrams = defaultdict(lambda:defaultdict(int))

for words in data:
    for i in range(len(words)-1):
        first_word = words[i] if words[i] in vocab else 'UNK'
        second_word = words[i + 1] if words[i+1] in vocab else 'UNK'
        bigrams[first_word][second_word] += 1

# turn the word counts into probabilities
for w in bigrams:
    bigrams[w] = dict(sorted(bigrams[w].items(), key=lambda item: -item[1]))
    total = sum(bigrams[w].values())
    bigrams[w] = dict([(k, bigrams[w][k]/total) for k in bigrams[w]])

# Applying Laplace Smoothing
V = len(vocab) + 1 # for UNK
laplace_smoothed_bigrams = defaultdict(lambda: defaultdict(float))
for w1 in bigrams:
  total_count = sum(bigrams[w1].values()) + V  # Adding V for Laplace smoothing
  for w2 in vocab | {'UNK'}:
    laplace_smoothed_bigrams[w1][w2] = round((bigrams[w1].get(w2, 0) + 1) / total_count, 5)

# Apply add k smoothing
# V = len(vocab) + 1 same as before
k = 0.01 # k value
k_smoothed_bigrams = defaultdict(lambda: defaultdict(float))
for w1 in laplace_smoothed_bigrams:
  total_count = sum(laplace_smoothed_bigrams[w1].values()) + k * V # add k smoothing
  for w2 in vocab | {'UNK'}:
    k_smoothed_bigrams[w1][w2] = round((laplace_smoothed_bigrams[w1][w2] + k) / total_count, 5)

#Take the ngram probabilities defined in the K-smoothed ngrams
#Input is the dictionary of probabilities
def PerplexityCalc(testData, probs, N):

  #Initialize the variable
  perpCalc = 0

  #Loop through the dictionary
  for words in testData:
    for i in range(len(words) - 1):

      #Add the -log of the n-gram probability of the word to perpCalc
      word1 = words[i]
      word2 = words[i + 1]
      if word1 not in probs:
        word1 = 'UNK'
      if word2 not in probs[word1]:
        word2 = 'UNK'
      perpCalc -= log(probs[word1][word2])

  #Divide by the total number of tokens
  perpCalc /= N

  #Return and exp the value
  return exp(perpCalc)

#Preprocess the validation (development) set
devData = preprocessing("val.txt")

#N is the total number of tokens in the test data
N = 0
for words in devData:
  N += len(words)

#Outputs
#Unigrams (each probability, and then the sum, respectively)

for unigram in unigrams:
    print(unigram, ":", unigrams[unigram])
print('Unigram Probability Total: ', sum(unigrams.values()))

#Bigrams and their probabilities with other words, then the sum of the values for each bigram
for key, value in bigrams.items():
    print(key, ':', value)
for w in bigrams:
    print(f'{w}|{bigrams[w]}| Sum: {sum(bigrams[w].values())}')

print()
#Run for Add-k smoothing
print("Add-K smoothing Perplexity:", PerplexityCalc(devData, k_smoothed_bigrams, N))

#Run for Laplace smoothing
print("Laplace smoothing Perplexity:", PerplexityCalc(devData, laplace_smoothed_bigrams, N))