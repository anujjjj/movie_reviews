
# coding: utf-8

# In[1]:


import nltk


# In[2]:


import sys
import sklearn


# In[19]:


nltk.download()


# In[5]:


from nltk.tokenize import sent_tokenize, word_tokenize

text = "Hello students, how are you doing today? The olympics are inspiring, and Python is awesome. You look nice today."

print(sent_tokenize(text))


# In[6]:


print(word_tokenize(text))


# In[7]:


from nltk.corpus import stopwords
print(set(stopwords.words('english')))


# In[8]:


example_sent = "This is some sample text, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

print(word_tokens)
print(filtered_sentence)
                     


# In[9]:


from nltk.stem import PorterStemmer

ps = PorterStemmer()

example_words = ["ride","riding","rider","rides"]

for w in example_words:
    print(ps.stem(w))


# In[10]:


# Now lets try stemming an entire sentence!

new_text = "When riders are riding their horses, they often think of how cowboys rode horses."

words = word_tokenize(new_text)

for w in words:
   print(ps.stem(w))


# In[13]:


nltk.download()


# In[14]:


# We can use documents from the nltk.corpus.  As an example, lets load the universal declaration of human rights.
from nltk.corpus import udhr
print(udhr.raw('English-Latin1'))


# In[18]:


# Lets import some sample and training text - George Bush's 2005 and 2006 state of the union addresses. 

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")


# In[20]:


# Now that we have some text, we can train the PunktSentenceTokenizer

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# Now lets tokenize the sample_text using our trained tokenizer

tokenized = custom_sent_tokenizer.tokenize(sample_text)


# In[21]:


# This function will tag each tokenized word with a part of speech

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))

        
# The output is a list of tuples - the word with it's part of speech
process_content()


# In[26]:



train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # combine the part-of-speech tag with a regular expression
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            # draw the chunks with nltk
            # chunked.draw()     

    except Exception as e:
        print(str(e))

        
process_content()


# In[27]:


# We can access the chunks, which are stored as an NLTK tree 

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # combine the part-of-speech tag with a regular expression
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)
            
            # draw the chunks with nltk
            # chunked.draw()     

    except Exception as e:
        print(str(e))

        
process_content()


# In[29]:


def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # The main difference here is the }{, vs. the {}. This means we're removing 
            # from the chink one or more verbs, prepositions, determiners, or the word 'to'.

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

            # chunked.draw()

    except Exception as e:
        print(str(e))

        
process_content()


# In[30]:


def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            # namedEnt.draw()
            
    except Exception as e:
        print(str(e))

        
process_content()


# In[31]:


import random
import nltk
from nltk.corpus import movie_reviews


# In[32]:


import random
import nltk
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# shuffle the documents
random.shuffle(documents)

print('Number of Documents: {}'.format(len(documents)))
print('First Review: {}'.format(documents[1]))

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

print('Most common words: {}'.format(all_words.most_common(15)))
print('The word happy: {}'.format(all_words["happy"]))


# In[34]:


print(len(all_words))
word_features = list(all_words.keys())[:4000]


# In[37]:


# The find_features function will determine which of the 3000 word features are contained in the review
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Lets use an example from a negative review
features = find_features(movie_reviews.words('neg/cv000_29416.txt'))
for key, value in features.items():
    if value == True:
        print (key)


# In[38]:


featuresets = [(find_features(rev), category) for (rev, category) in documents]


# In[39]:


# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# define a seed for reproducibility
seed = 1

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)


# In[40]:


print(len(training))
print(len(testing))


# In[41]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))

