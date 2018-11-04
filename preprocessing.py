"""
Copyright Xavier van Egdom
"""

from random import shuffle

# make a nice list of the dataset
def get_dataset_list(name):
    # load the datasets
    with open(name, 'r') as f:
        dataset = f.read()
    
    tempset, temprow, temp = [], [], ''
    
    for i in dataset:
        
        if i is ';':
            temprow.append(temp)
            temp = ''
            
        elif i is "\n":
            temprow.append(temp)
            tempset.append(temprow)
            temprow, temp = [], ''
        
        else:      
            temp += i
    
    temprow.append(temp)
    tempset.append(temprow)
    return tempset

def get_X(begin, end, dataset):
    templist = []
    for i in dataset:
            templist.append(i[begin:end])
    make_all_float(templist)
    return templist

def make_all_float(dataset):
    i = 0
    while i < len(dataset):
        j = 0 
        while j < len(dataset[0]):
            try:
                dataset[i][j] = float(dataset[i][j])
            except ValueError:
                dataset[i][j] = dataset[i][j]
            j += 1
        i += 1
            

def get_Y(index, dataset):
    templist = []
    for i in dataset:
            templist.append(int(i[index]))
    return templist


def label_categorical(dataset, index):
    categories = []
    for row in dataset:
        categories.append(row[index])
    
    categories = set(categories)
    categories = list(categories)
    
    for row in dataset:
        row[index] = categories.index(row[index]) 

# x norm = x - min(x)  / max(x) - min(x)        
def normalisation(dataset):
    templist = [ [] for i in range(len(dataset))]
    
    for i in range(len(dataset[0])):
        tempcolumn = []
        for row in dataset:
            tempcolumn.append(row[i])
        
        j = 0
        while j < len(dataset):
            templist[j].append(normalisation_min_max(tempcolumn[j], tempcolumn))
            j += 1
        
    return templist

# dont call this one            
def normalisation_min_max(x, dataset):
    return (x - min(dataset)) / (max(dataset) - min(dataset))

# shuffle the sets
def shuffle_sets(X, y):    
    list_to_shuffle = []
    #join the list
    for i in range(len(X)):
        temp = (X[i],y[i])
        list_to_shuffle.append(temp)
    
    shuffle(list_to_shuffle)
    #disjoin the list
    X = []
    y = []
    for item in list_to_shuffle:
        X.append(item[0])
        y.append(item[1])
    return X, y 

#make training set and test set
def train_test_split(X, y, training_size):
    # slice training and test X set
    X_train = X[:int(len(X) * training_size)]
    X_test = X[int(len(X) * training_size):]
    # slice training and test y set
    y_train = y[:int(len(y) * training_size)]
    y_test = y[int(len(y) * training_size):]
       
    return X_train, X_test, y_train, y_test