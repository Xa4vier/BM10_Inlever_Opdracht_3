def save_probabilities(name, dataset):
    with open(f'{name}.csv', 'w', encoding='latin-1') as f:
        for line in dataset:
            f.writelines(f'{line}\n') 

def save_gram(name, dataset):
    with open(f'{name}.csv', 'w', encoding='latin-1') as f:
        for line in dataset:
            for gram in line:
                f.write(f'{gram},') 

def save_set_gram_p(name, dataset):
        save_gram(f'{name}_gram', dataset[0])
        save_probabilities(f'{name}_p', dataset[1])
        
def save_grams(names, name, grams):          
    for i in range(len(names)):
        save_set_gram_p(f'grams/{name}_{names[i]}', grams[i])