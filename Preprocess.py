import re


## Read in text/data
with open('waseemtrain.txt', 'r') as tr:
    train = tr.readlines()

#with open('waseemtest.txt', 'r') as te:
    #test = te.readlines()

## Remove URLs and save tweets
clean_train = []
#clean_test = []

for i in train:
    trainr = re.sub(r'http\S+', '', i)
    clean_train.append(trainr)

#for i in test:
    #testr= re.sub(r'http\S+', '', i)
    #clean_test.append(testr)

## write cleaned text into a new file
with open('cleantrain.txt', 'w') as train:
    for tweet in clean_train:
        train.write(tweet)

#with open('cleantest.txt', 'w') as test:
    #for tweet in clean_test:
        #test.write(tweet)
