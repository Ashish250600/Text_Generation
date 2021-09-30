#Import the required dependencies
import sys
import numpy
import nltk
import tensorflow
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#Load the data
file=open("frankenstein.txt",encoding="utf8").read()

#Tokenization/Standardization
def tokenize_words(input):
    input=input.lower()
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(input)
    filtered=filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)
processed_inputs=tokenize_words(file)

#Char to numbers
chars=sorted(list(set(processed_inputs)))
char_to_num=dict((c,i) for i,c in enumerate(chars))

#Check if words to chars or char to num has worked?
input_len=len(processed_inputs)
vocab_len=len(chars)
print("Total number of characters: ", input_len)
print("Total vocab: ",vocab_len)

#Seq length
seq_length=100
x_data=[]
y_data=[]

#Loop through the sequence
for i in range (0, input_len - seq_length, 1):
    in_seq= processed_inputs[i:i + seq_length]
    out_seq= processed_inputs [i + seq_length]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])                 
n_patterns=len(x_data)
print ("Total Patterns: ", n_patterns)

#Convert the input sequence to np aray and so on
x=numpy.reshape(x_data, (n_patterns, seq_length, 1))
x=x/float(vocab_len)

#One-hot encoding
y=np_utils.to_categorical(y_data)

#Creating the model
model=Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0,2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Saving weights
filepath="model_weights_saved.hdf5"
checkpoint=ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks= [checkpoint]

#Fit model and let it train
model.fit(x,y, batch_size=256, epochs=4, callbacks=desired_callbacks)

#Recompile model with saved weigths
filename="model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')

#Output of the model back into characters
num_to_char=dict((i,c) for i,c in enumerate(chars))

#Random seed to help generate
start=numpy.random.randint(0, len(x_data)-1)
pattern=x_data[start]
print("Randon seed: ")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

#Generate the text
for i in range(1000):
    x=numpy.reshape(pattern, (1,len(pattern), ))
    x=x/float(vocab_len)
    prediction=model.predict(x, verbose=0)
    index=numpy.argmax(prediction)
    result=num_to_char[index]
    seg_in=[num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern=pattern[1:len(pattern)]


        
     
