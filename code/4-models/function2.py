###############################################################################
!python -m spacy download en_core_web_md &> /dev/null

import pandas as pd
import numpy as np
import nltk
import re
import itertools
import matplotlib.pyplot as plt

## nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag

# spacy
import spacy
# md: reduced word vector table with 20k unique vectors for ~500k words
nlp = spacy.load("en_core_web_md") # IF THIS DOESN'T WORK, THEN RUN THE CODE ABOVE^ & RE-START RUNTIME

# Import label encoder
from sklearn import preprocessing

# create function for preprocessing including word embedding

def create_y(df):
    y_train = df["label"]
    # Encode labels in column 'label'.
    y_train = label_encoder.fit_transform(y_train)
    
    return y_train
    

def create_x(df):
    
    try:
        X_train = df["tweet"]
    try:
        X_train = df["statement"]
    
    X_train = X_train.map(lambda x: CleanText(x))
   
 
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()


    # generate word vectors for each tweet 
    ## train set 
    tweet2vec_list = [nlp(doc).vector.reshape(1,-1) for doc in X_train] # creates 6420x1 list, with each entry containing a 300x1 np.array word vector corresponding with a tweet
    tweet2vec_data = np.concatenate(tweet2vec_list) # joins word vectors into 6420x300 np.array
    tweet2vec_train = pd.DataFrame(tweet2vec_data) # convert to data frame

    # count vectorizer
    cv = CountVectorizer(ngram_range=(1, 2)) # count term frequency

    # fit and transform train data to count vectorizer
    cv.fit(X_train.values)
    cv_train = cv.transform(X_train.values)

    ## train set
    ### create list of word embedding column names
    word2vec_col = []
    for i in range(len(tweet2vec_train.columns)):
      num = str(i)
      name = "word2vec_"+num
      word2vec_col.append(name) 

    ### rename word embedding columns 
    tweet2vec_train.columns = word2vec_col
    
    # TF-IDF
    tfidf = TfidfTransformer()

    # fit the CountVector to TF-IDF transformer
    tfidf.fit(cv_train)
    tfidf_train = tfidf.transform(cv_train)
    
    # convert tfidf_train to data frame
    ## train set
    tfidf_train = pd.DataFrame(tfidf_train.toarray())
    
    
    # rename tfidf columns to de-conflict merge

    ## train set
    ### create list of tfidf column names
    tfidf_col = []
    for i in range(len(tfidf_train.columns)):
      num = str(i)
      name = "tfidf_"+num
      tfidf_col.append(name) 

    ### rename tfidf columns
    tfidf_train.columns = tfidf_col
    
    # join tf-idf with word embeddings 

    ## train set 
    X_train = tfidf_train.join(tweet2vec_train) 
    
    return X_train






# import dependencies
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
stoplist = set(stopwords.words('english')) # create stop word list
lemmatizer = WordNetLemmatizer() # initializer lemmatizer 

# create clean_text() function
def clean_text(string):
    text = string.lower() # lowercase
    text = re.sub(r"http(\S)+",' ',text) # remove URLs   
    text = re.sub(r"www(\S)+",' ',text) # remove URLs
    text = re.sub(r"&",' and ',text) # replace & with ' and '
    text = text.replace('&amp',' ') # replace &amp with ' '
    text = re.sub(r"[^0-9a-zA-Z]+",' ',text) # remove non-alphanumeric characters
    text = text.split() # splits into a list of words
    text = [w for w in text if not w in stoplist] # remove stop words
    text = [lemmatizer.lemmatize(w) for w in text] # lemmatization
    text = " ".join(text) # joins the list of words
    return text
    

###############################################################################
## import dependencies
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# define print_metrics() function to print results
def print_metrics(pred,true):
    print(confusion_matrix(true,pred))
    print(classification_report(true,pred,))
    print("Accuracy : ",accuracy_score(pred,true))
    print("Precison : ",precision_score(pred,true, average = 'weighted'))
    print("Recall : ",recall_score(pred,true, average = 'weighted'))
    print("F1 : ",f1_score(pred,true, average = 'weighted'))


###############################################################################
## import depedencies
import matplotlib.pyplot as plt
import numpy as np
import itertools

# define plot_confusion_matrix() function to display results
def plot_confusion_matrix(cm,
                        target_names,
                        title='Confusion matrix',
                        cmap=None,
                        normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels,            # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label')
    plt.show()