from transformers import ElectraTokenizer
import numpy as np
import tokenization

def RepresentsInt(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


vocab = tokenization.load_vocab(vocab_file='vocab.txt')
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

#vocab = tokenization.load_vocab(vocab_file='html_mecab_vocab_128000.txt')
#tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

count = 0
#tokenizer_ = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

triple_file = open('wellness.n-triples', 'r', encoding='utf-8')
triple_lines = triple_file.read().split('\n')

eval_file = open('triples_for_medical_knowledge', 'w', encoding='utf-8')

words = []
labels = []

for line in triple_lines:
    if line.find('#label') != -1:
        tk = line.split('>')

        object_triple = tk[0].split('/')[-1].replace('_', ' ')
        relation_triple = tk[1].split('/')[-1]
        subject_triple = tk[2].replace('\"', '').replace('.', '').replace('@ko', '')

        words.append(object_triple)
        labels.append(subject_triple)

words = np.array(words, dtype='<U100')
labels = np.array(labels, dtype='<U100')

arg_idx = words.argsort()
words.sort()

for line in triple_lines:
    if len(line) == 0:
        continue

    if line.find('#label') != -1 or line.find('-rdf-syntax-n') != -1:
        continue

    tk = line.split('>')

    object_triple = tk[0].split('/')[-1].replace('_', ' ')

    idx = words.searchsorted(object_triple)
    try:
        if words[idx] == object_triple:
            object_triple = labels[arg_idx[idx]]

            relation_triple = tk[1].split('/')[-1]
            subject_triple = tk[2].replace('\"', '').replace('.', '')
    except:
        continue

    tokens = tokenizer.tokenize(subject_triple)
    if len(tokens) == 1:
        eval_file.write(object_triple + '[split]' + relation_triple + '[split]' + subject_triple + '\n')
        count += 1

print(len(triple_lines))
print(count)

eval_file.close()
triple_file.close()