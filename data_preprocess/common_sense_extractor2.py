from transformers import ElectraTokenizer
import tokenization


def RepresentsInt(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


#vocab = tokenization.load_vocab(vocab_file='vocab.txt')
#tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

count = 0
vocab = tokenization.load_vocab(vocab_file='html_mecab_vocab_128000.txt')
tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

triple_file = open('nia_common_triples@180105.nt', 'r', encoding='utf-8')
triple_lines = triple_file.read().split('\n')

eval_file = open('triples_for_koelectra', 'w', encoding='utf-8')

for line in triple_lines[0:600000]:
    if len(line) == 0:
        continue

    tk = line.split('>')

    object_triple = tk[0].split('/')[-1].replace('_', ' ')
    relation_triple = tk[1].split('/')[-1]
    subject_triple = tk[2].replace('\"', '').replace('.', '')

    tokens = tokenizer.tokenize(subject_triple)

    eval_file.write(object_triple + '[split]' + relation_triple + '[split]' + subject_triple + '\n')
    count += 1

print(len(triple_lines))
print(count)

eval_file.close()
triple_file.close()