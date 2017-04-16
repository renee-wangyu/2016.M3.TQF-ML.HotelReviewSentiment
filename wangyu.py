# -*- coding: utf-8 -*-
"""

"""
import os
import itertools
import jieba
import re
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc

def load_from_txt(fdir, stop_words):
    """loading data from txt 

    @type fdir: str
    @type stop_words: set
    @rtype: data List
    @rtype: word_set Dict
    """
    
    data = []
    word_set = set()
    p1 = re.compile('[0-9a-zA-Z]')
    if os.path.exists(fdir) == False:
        print (fdir, 'does not exist, try again!')
        return None
    for fname in os.listdir(fdir):
        content = []
        for line in open(fdir+'/'+fname,encoding='gb18030',errors='ignore'):
            line = p1.subn('',line)[0]
            for word in jieba.cut(line.strip()):
                if len(word) > 1 \
                   and word not in stop_words:
                    content.append(word)
                    word_set.add(word)
        data.append(' '.join(content))
    return data, word_set



def write_dict(word_set, fname):
    """write the dict into a file

    @type word_set: word dictionary
    @type fname: dictionary filename
    """
    f = open(fname,'w',encoding='utf8')
    for word in word_set:
        #use the utf8 encode
        f.write(word+'\n')
    f.close()
    
def write_seg_result(data, fname):
    """write the seg result into a file

    @type data: seg data
    @type fname: seg filename
    """
    f = open(fname,'w',encoding='utf8')
    for line in data:
        #use the utf8 encode
        f.write(line+'\n')
    f.close()

def write_result(result_data, fname):
    """write the result into a file

    @type data: reuslt data
    @type fname: result filename
    """
    f = open(fname,'w',encoding='utf8')
    for d in result_data:
        s = ''
        for dd in d[:-1]:
            s += str(dd)+','    
        f.write(s+str(d[-1])+'\n')
    f.close()

def word_count(data, word_set):
    """count the data into the map
    @type data: seg data
    @type word_set: word dictionary
    """
    wd = dict(zip(word_set, range(len(word_set))))
    result_data = []
    for line in data:
        d = [0]*len(word_set)
        for word in line.split(' '):
            if len(word) > 0:
                d[wd[word]] += 1
        result_data.append(d)
    return result_data

def get_data_result(fdir):
    print ('load stop words')
    stop_words = set([i.strip() for i in open('stoplis.txt',encoding='utf8')])
    print ('load file and seg:')    
    data,word_set = load_from_txt(fdir, stop_words)
    print ('seg over')
    print ('write the dict')
    write_dict(word_set, 'my_dict_'+fdir+'.txt')
    print ('write the seg file')
    write_seg_result(data, 'seg_'+fdir+'.txt')
    print ('start to word count')    
    return data,word_set

def text_preocess():
    data = [i.strip() for i in open('seg_pos.txt',encoding='utf8')]    
    label_pos = np.ones((3000,1),dtype='int')
    label_neg = np.zeros((3000,1),dtype='int')
    label = np.vstack((label_pos,label_neg))    
    data = data+[i.strip() for i in open('seg_neg.txt',encoding='utf8')]
    c = CountVectorizer(min_df=3,ngram_range=(1,3))
    tfidf = TfidfTransformer()
    tfidf_all=tfidf.fit_transform(c.fit_transform(data))
    print(tfidf_all.shape)
    data = tfidf_all
    labels = label
    return data.toarray(), labels.reshape((labels.shape[0],))

def get_train_test_split(data, labels, model, model_name,my_test_size=0.2):
    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=my_test_size)    
    model.fit(xtrain, ytrain)
    ypredict = model.predict(xtest)
    print ('accuracy is:',accuracy_score(ytest, ypredict))
    cnf_matrix = confusion_matrix(ytest, ypredict)
    np.set_printoptions(precision=2)
    # Plot normalized confusion matrix
    f = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["negtive","positive"], normalize=True,
                          title='confusion matrix:'+model_name)
    plt.savefig(model_name+"_cm.png")
    #plt.show()
    #print (ytest.shape,ypredict.shape)
    plot_roc(ytest,ypredict,model_name)
    #plot_roc(ytest.reshape(ytest.shape[0],1), ypredict.reshape(ypredict.shape[0],1),model_name)

def plot_roc(y_test, y_score,model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print ("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(model_name+"_roc.png")
    #plt.show()
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, lw=1)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(model_name+"_roc.png")
    plt.show()
    '''


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_valid_score(data, labels, model,cv_size = 5):
    metric = cross_val_score(model,data,labels,cv=cv_size,scoring='precision').mean()
    print ('cross valid score is:',metric)

def get_nb_result(data,labels):
    print ('---------naive bayes result is:---------')
    model = GaussianNB()
    get_train_test_split(data, labels, model,'nb')
    model = GaussianNB()
    get_valid_score(data, labels, model)

def get_lr_result(data,labels):
    print ('---------Logistic Regression result is:---------')
    model = LogisticRegression()
    get_train_test_split(data, labels, model,'lr')
    model = LogisticRegression()
    get_valid_score(data, labels, model)

def get_dt_result(data,labels):
    print ('---------decision tree classifier result is:---------')
    model = DecisionTreeClassifier()
    get_train_test_split(data, labels, model,'dt')
    model = DecisionTreeClassifier()
    #get_valid_score(data, labels, model)

def get_svm_result(data,labels):
    print ('---------svm result is:---------')
    model = SVC(kernel='linear')
    get_train_test_split(data, labels, model,'svm')
    model = SVC(kernel='linear')
    #get_valid_score(data, labels, model)

def predict():
    pos_data = np.array([i.strip().split(',') for i in open('data_pos.txt')])
    m1 = np.array(pos_data,dtype='int')
    label = np.ones((3000,1),dtype='int')
    pd = np.hstack((m1,label))
    #print(pd.shape)
    neg_data = np.array([i.strip().split(',') for i in open('data_neg.txt')])
    m1 = np.array(neg_data,dtype='int')
    label = np.zeros((3000,1),dtype='int')
    nd = np.hstack((m1,label))
    print (pd.shape)
    print (nd.shape)
    ad = np.concatenate((pd,nd))
    data = np.array(ad[:, :-1])
    
    labels = np.array(ad[:,-1])
    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.4, random_state=1)
    model = GaussianNB()
    model.fit(xtrain, ytrain)
    ypredict = model.predict(xtest)
    print ('accuracy is:',accuracy_score(ytest, ypredict))

if __name__ =='__main__':
    # main function    
    '''
    posdata,word_set_pos = get_data_result('pos')
    negdata,word_set_neg = get_data_result('neg')
    word_set = word_set_pos | word_set_neg
    result_data = word_count(posdata, word_set)
    write_result(result_data, 'data_pos.txt')
    result_data = word_count(negdata, word_set)
    write_result(result_data, 'data_neg.txt')
    '''
    data,labels = text_preocess()
    get_nb_result(data,labels)
    get_dt_result(data,labels)
    get_lr_result(data,labels)
    get_svm_result(data,labels)
    #predict()