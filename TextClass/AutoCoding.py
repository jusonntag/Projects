import time
start_time = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
import gensim
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

############################################# functions for text processing #######################################################################

def text_cleaning(df, for_embedding = False, manual_manipulation = False):
    
    # define stemmer & stop word for following functions
    stemmer = SnowballStemmer('german')
    stop_words = set(stopwords.words('german'))

    ger_punctuation = [',', '.', '!', '?', '/' , ':', ';', '-', '_', "'", '"', '(', ')', '+', '&', '=', 'quot', '“', '”', '„']

    def count_punct(text):
        count = sum([1 for char in text if char in ger_punctuation])
        return round(count/(len(text) - text.count(' ')),3)*100

    body_len = df[0].apply(lambda x: len(x) - x.count(' '))
    punct_per = df[0].apply(lambda x: count_punct(x))

    lst_words = list()
    lst_sentence = list()
    text_clean = ''
    text_for_dic = list()

    for i in range(len(df)):
        if isinstance(df, pd.core.series.Series):
            line_of_data = df.loc[i]
        else:
            line_of_data = df.loc[i,0]

        word_tokens = word_tokenize(line_of_data)


        if for_embedding:
            words_filtered = word_tokens
            words_filtered = [word for word in words_filtered if word not in ger_punctuation]
            lst_words.append(words_filtered)
            text_clean = ' '.join(words_filtered)
            lst_sentence.append(text_clean)
        else:
            #lowering, stemming and removing stop words
            word_tokens = [word.lower() for word in word_tokens]
            words_filtered = [word for word in word_tokens if word not in stop_words] # rethink stemming --> stemmer.stem()
            words_filtered = [word for word in words_filtered if word not in ger_punctuation]
            
            #words_filtered = [word for word in words_filtered if word ]

            # own stemming or rather 'replacements'
            if manual_manipulation:
                words_filtered = [word.replace('pat', 'patient') for word in words_filtered]
                words_filtered = [word.replace('patienten', 'patient') for word in words_filtered]

                words_filtered = [word.replace('gute', 'gut') for word in words_filtered]
                words_filtered = [word.replace('gutr', 'gut') for word in words_filtered]
                words_filtered = [word.replace('gutn', 'gut') for word in words_filtered]
                words_filtered = [word.replace('guts', 'gut') for word in words_filtered]

                words_filtered = [word.replace('neue', 'neu') for word in words_filtered]
                words_filtered = [word.replace('neuen', 'neu') for word in words_filtered]
                words_filtered = [word.replace('neun', 'neu') for word in words_filtered]

                words_filtered = [word.replace('studien', 'studie') for word in words_filtered]

                words_filtered = [word.replace('informationen', 'information') for word in words_filtered]
                words_filtered = [word.replace('infos', 'information') for word in words_filtered]
                words_filtered = [word.replace('info', 'gut') for word in words_filtered]

                words_filtered = [word.replace('wichtige', 'wichtig') for word in words_filtered]

                words_filtered = [word.replace('wirksamkeitgut', 'wirksamkeit gut') for word in words_filtered]
                words_filtered = [word.replace('datengut', 'daten gut') for word in words_filtered]
                words_filtered = [word.replace('gespraechgut', 'gespraech gut') for word in words_filtered]

                words_filtered = [word.replace('zulassen', 'zulassung') for word in words_filtered]
                words_filtered = [word.replace('zugelassen', 'zulassung') for word in words_filtered]

                words_filtered = [word.replace('koennen', 'kann') for word in words_filtered]
                words_filtered = [word.replace('koennte', 'kann') for word in words_filtered]
                words_filtered = [word.replace('koennten', 'kann') for word in words_filtered]

                #eleminating uninformative words
                #eleminating specific stop words? not done yet

                words_filtered = [word.replace('fuer', '') for word in words_filtered]
                words_filtered = [word.replace('beim', '') for word in words_filtered]
                words_filtered = [word.replace('per', '') for word in words_filtered]
                words_filtered = [word.replace('ueber', '') for word in words_filtered]
                words_filtered = [word.replace('gemacht', '') for word in words_filtered]
                words_filtered = [word.replace('os', '') for word in words_filtered]
                words_filtered = [word.replace('per', '') for word in words_filtered]

            else:
                words_filtered = [stemmer.stem(word) for word in words_filtered]
                words_filtered = [word.replace('fuer', '') for word in words_filtered]


            lst_words.append(words_filtered)
            text_clean = ' '.join(words_filtered)
            text_for_dic = text_for_dic+words_filtered
            lst_sentence.append(text_clean)
    
    all_words = Counter(text_for_dic)
    word_features =list(all_words.keys())[:int(len(all_words)*0.05)] # ~5% of vocab
    
    return lst_sentence, lst_words, text_clean, body_len, punct_per, word_features
    

def target_cleaning(df, for_embedding = False, manual_manipulation = False):
    ger_punctuation = [',', '.', '!', '?', '/' , ':', ';', '-', '_', "'", '"', '(', ')', '+', '&', '=', 'quot', '“', '”', '„']

    lst_words = list()
    lst_sentence = list()
    text_clean = ''

    for i in range(len(df)):
        if isinstance(df, pd.core.series.Series):
            line_of_data = df.loc[i]
            
            word_tokens = line_of_data.split(' ')

            if for_embedding:
                words_filtered = word_tokens
                lst_words.append(words_filtered)
                text_clean = ' '.join(words_filtered)
                lst_sentence.append(text_clean)
            else:
                #lowering, stemming and removing stop words
                words_filtered = [word.lower() for word in word_tokens]
                words_filtered = [word.replace('(sehr)', 'sehr') for word in words_filtered]
                words_filtered = [word.replace('(neue)', 'neue') for word in words_filtered]
                words_filtered = [word.replace('fuer', '') for word in words_filtered]
                words_filtered = [word for word in words_filtered if word not in ger_punctuation]
                
                print(words_filtered)

                lst_words + ' '.join(words_filtered)
                text_clean = ' '.join(words_filtered)
                lst_sentence.append(text_clean)

        text_clean = text_clean + ' '.join(words_filtered)

    return lst_sentence, lst_words, text_clean



############################################# data preprocessing ##################################################################################
# load features - words
df = pd.read_excel(os.getcwd()+'/TextClass/Copy of AutoCodeData.xlsx', header=None)
df_numbers_of_codes = df
# load targets
df_code = pd.read_excel(os.getcwd()+'/TextClass/Copy of Codes.xlsx', header=None)

# creating a df with words as codes instead of numbers
for i in df[1]:
    for j in df_code[0]:
        if i == j:
            index = df_code[0].index[df_code[0] == j][0]
            code_string = df_code.loc[index,1]
            df = df.replace(to_replace=j, value = code_string)
df_target = df[1]
df_target = df_target.fillna('')

# drop nan and no used columns from targets
df_numbers_of_codes = df_numbers_of_codes.drop([2,3],axis =1)
df_numbers_of_codes = df_numbers_of_codes.dropna()

# remove codings with a frequency below 60
value_counts = df_numbers_of_codes[1].value_counts()
numbers_to_drop = value_counts[value_counts < 60].index                                      # change number here for the least number of samples per label
df_numbers_of_codes = df_numbers_of_codes[~df_numbers_of_codes[1].isin(numbers_to_drop)] 

# reset index // y is final target df
df_numbers_of_codes.reset_index(drop=True, inplace = True)
y=df_numbers_of_codes[1]


fre = Counter(y)
print(len(y), len(fre))
for number, count in fre.items():
    print(f"Number {number}: Frequency {count}")
print('Size of dataset:',len(fre), 'different labels')

# clean words and bring them into format for TD-IDF and Word2Vec
vec_sen, words, text, _, _, word_features = text_cleaning(df_numbers_of_codes, for_embedding=False, manual_manipulation=False) # returns tokens as sentences, Tokens as words and one variable containing ALL tokens as one string
vec_sen_code2num, words_code2num, text_code2num, body_len, punct_per, _ = text_cleaning(df_numbers_of_codes,for_embedding=True,manual_manipulation=False)

################################################ TF-IDF + train_test_split ###################################################################

# lower all words
words_code2num = [[word.lower() for word in sublist] for sublist in words_code2num]
flat_word2vec_list = [word for sublist in words_code2num for word in sublist] # flattening the nested list

# word2vec_lsit/df is a list/df which contains every word once - like a vocabulary
word2vec_list = set(flat_word2vec_list)
word2vec_df = pd.DataFrame(words_code2num)

# OneHotEncoding the words of each sentence
vectorizer = CountVectorizer(binary=True)
one_hot_encoded = vectorizer.fit_transform([' '.join(word) for word in words])
one_hot_encoded = one_hot_encoded.toarray()
df_onehot = pd.DataFrame(one_hot_encoded)

# OneHotEncoding the words of each sentence for embedding
vectorizer = CountVectorizer(binary=False)
one_hot_embedding = vectorizer.fit_transform([' '.join(word) for word in words_code2num])
one_hot_embedding = one_hot_embedding.toarray()

# makes one big df for all applied different preprocessing techniques
df_numbers_of_codes = pd.DataFrame({'Sentence': vec_sen})
df_numbers_of_codes = pd.concat([df_numbers_of_codes, df_onehot], axis=1)

y = y.rename('target')

X_train, X_test, y_train, y_test = train_test_split(df_numbers_of_codes,y, test_size=0.15, random_state=42, stratify = y)

#apply TD-IDF
tfidf_vec = TfidfVectorizer(ngram_range=(2,4), norm='l2') # norm = None
tfidf_vec_fit = tfidf_vec.fit(X_train['Sentence'])
tfidf_train = tfidf_vec_fit.transform(X_train['Sentence'])
tfidf_test = tfidf_vec_fit.transform(X_test['Sentence'])

#drop coloumn 'sentance' because all data is now encoded as one hot
X_train = X_train.drop('Sentence', axis=1)
X_test = X_test.drop('Sentence', axis=1)

# train and test features encoded as tfidf
X_train_vec = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(tfidf_train.toarray())], axis=1)
X_train_vec.columns = X_train_vec.columns.astype(str)
X_test_vec = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(tfidf_test.toarray())], axis=1)
X_test_vec.columns = X_test_vec.columns.astype(str)

################################################ Word2Vecort #################################################################    

# define Word2Vec model with a output vectorsize of 20
model = gensim.models.Word2Vec(
    vector_size = 100,
    window = 4,
    min_count = 1,
    workers = 4
)

# training the model
model.build_vocab(words, progress_per=1000)
model.train(words, total_examples=model.corpus_count, epochs=model.epochs)

# crating a dict where each word is the key and the w2vector is the value
vocabulary = model.wv.key_to_index
embedded_vocab = [model.wv[word] for word in vocabulary]
vectorized_vocabulary = dict(zip(vocabulary, embedded_vocab))

# recreates the sentences where each word is replaced by ist vector
vectorized_sentences = []
for sentence in words:
    vectorized_sentence = []
    for item in sentence:
        vector = vectorized_vocabulary[item]
        vectorized_sentence.extend(vector.tolist())
    vectorized_sentences.append(vectorized_sentence)

# padding the sequences to the same length
max_length = len(max(vectorized_sentences, key=len))
padded_sentences = [sentence + [0] * (max_length - len(sentence)) for sentence in vectorized_sentences]

w2v_df = pd.DataFrame(padded_sentences)
w2v_df = w2v_df.fillna(0)
# normalize data
w2v_df /= w2v_df.max()

X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(w2v_df,y, test_size=0.15, random_state=42, stratify = y)

################################################ SVM ###################################################################

# train a SVM on TFIDF data
svm_clf = svm.SVC(kernel='linear', gamma=0.1, C =1)
svm_clf.fit(X_train_vec,y_train)
y_pred = svm_clf.predict(X_test_vec)
accuracy = accuracy_score(y_test,y_pred)
print('SVM accuracy TD-IDF:', accuracy)


################################################ Naive Bayes ###################################################################

# define grid search parameters
param_grid = {'alpha':[0.001,0.01,0.1,0.5,1.0,1.5,2.0]}

# define Naive Bayes Class
nb_clf = naive_bayes.MultinomialNB()
grid_search = GridSearchCV(nb_clf, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train_vec,y_train)

print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Validation Score: ", grid_search.best_score_)

# evaluate the Naive Bayes classifier with the best hyperparameters on the validation set
best_clf = grid_search.best_estimator_
validation_score = best_clf.score(X_test_vec, y_test)
print("Validation Score with Best Hyperparameters: ", validation_score)
y_pred = best_clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

########################################## 1. Neural network Tfidf #################################################################
# train simple neural network with TF-IDF data

neural = True
if neural == True:
    #get shape of input layer
    input_layer = X_train_vec.shape[1]
    #get shape of output layer  -> number of classes
    output_layer = len(fre)
    y_train = pd.DataFrame({'class':y_train})
    y_test = pd.DataFrame({'class':y_test})


    # prepare train & test targets for neural network -> one hot encoded
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(y_train[['class']])
    encoded = encoded_data.toarray()
    y_train = np.array(encoded)
    y_test_encoded = encoder.transform(y_test[['class']])
    y_test_encoded = y_test_encoded.toarray()
    y_test = np.array(y_test_encoded)

    # define neural network
    nn = Sequential()
    nn.add(Dense(15, activation = 'relu', input_shape = (input_layer,)))
    nn.add(Dense(45, activation = 'relu'))
    nn.add(Dense(output_layer, activation = 'softmax'))

    nn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])


    assert X_train_vec.shape[0] == y_train.shape[0], "Number of samples in X_test_vec and y_train_encoded do not match."
    history = nn.fit(X_train_vec, y_train, epochs = 45, batch_size = 30, validation_data=(X_test_vec,y_test))


    test_loss, test_acc = nn.evaluate(X_test_vec, y_test)
    print('Test accuracy of FF with Tfidf:', test_acc)
    nn.summary()

    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Loss Function Over Epochs')
    plt.legend()
    plt.xlabel('No. epoch')
    plt.grid(True)
    plt.show()

    pd.DataFrame(history.history).plot(figsize=(10,10))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

end_time = time.time()
print('time needed:',end_time - start_time, 's')


