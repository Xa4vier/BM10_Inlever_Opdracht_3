def load_probabilities(name):
    file = []
    with open(f'{name}_p.csv', 'r', encoding='latin-1') as f:
        for line in f:
            file.append(float(line))
    return file

def load_gram(name):
    file = ''
    with open(f'{name}_gram.csv', 'r', encoding='latin-1') as f:
        for line in f:
            file += (line)
    return file

def load_ngram(name, n):
    dataset = load_gram(name)
    dataset = dataset.split(',')
    templ = []
    temp = []
    for i in range(0, len(dataset) - n, n):
        temp = []
        for j in range(i, i + n):
            temp.append(dataset[j])
        templ.append(temp)
    return templ

def load_chain(name, n):
    return [load_ngram(name, n), load_probabilities(name)]