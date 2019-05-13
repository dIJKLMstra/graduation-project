'''
    @author Qi Sun
    @desc using classifiers to predict the results
'''

import os
import sys
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

vector_path = '../data/ft_vec'
pickle_path = '../data/clf_pickles'
result_path = '../data/result'

def svm_predict():
    para_data = '../data/ft_vec/para_vec.tsv'
    meta_data = '../data/ft_vec/meta_vec.tsv'

    with open(para_data, 'r') as pd:
        pd_lines = pd.readlines()
    pd_lines = [line[:-1].split('\t') for line in pd_lines]

    with open(meta_data, 'r') as md:
        md_lines = md.readlines()
    md_lines = [line[:-1].split('\t') for line in md_lines]

    all_data = pd_lines + md_lines

    random.shuffle(all_data)

    train_cnt = int(len(all_data)*0.8)
    train_data = [vec[:-1] for vec in all_data[:train_cnt]]
    train_target = [vec[-1] for vec in all_data[:train_cnt]]
    #print(train_data)
    #print(train_target)

    test_data = [vec[:-1] for vec in all_data[train_cnt:]]
    test_target = [vec[-1] for vec in all_data[train_cnt:]]

    clf = svm.SVC()
    clf.fit(train_data, train_target)

    print(clf.score(test_data, test_target))

def classifiers_fit():
    '''
        using different classifiers' pickles
        @command example
        python svm_pre.py train_vector_file test_vector_file pickle_prefix
    '''

    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    pickle_prefix = sys.argv[3]
    
    train_vec = os.path.join(vector_path, trainFile)
    test_vec = os.path.join(vector_path, testFile)

    with open(train_vec, 'r') as trainF:
        train_lines = trainF.readlines()

    with open(test_vec, 'r') as testF:
        test_lines = testF.readlines()

    train_vecs = [line[:-1].split('\t') for line in train_lines]
    test_vecs = [line[:-1].split('\t') for line in test_lines]
    x_train = [line[:-1] for line in train_vecs]
    y_train = [line[-1] for line in train_vecs]
    x_test = [line[:-1] for line in test_vecs]
    x_set = x_train + x_test

    # we need to normalize vector first to avoid influence of difference
    scaler = StandardScaler()
    fited_scaler = scaler.fit(x_set)
    x_train = fited_scaler.transform(x_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_scaler.pkl'), 'wb') as fs:
        pickle.dump(fited_scaler, fs)

    # Support Vector Machine ##
    print("SVM.........")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_svm.pkl'), 'wb') as svmc:
        pickle.dump(clf, svmc)

    # Logisitic Regression classifier ##
    print("LogisticRegression........")
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_lr.pkl'), 'wb') as lr:
        pickle.dump(clf, lr)

    # Random Forest classifier ##
    print("RandomForest.........")
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_rf.pkl'), 'wb') as rf:
        pickle.dump(clf, rf)

    # KNN ###
    print("KNN..........")
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_knn.pkl'), 'wb') as knn:
        pickle.dump(clf, knn)

    # Decision Tree ###
    print("Decision Tree...........")
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_dt.pkl'), 'wb') as dt:
        pickle.dump(clf, dt)

    # Gradient Boosting ##
    print("GradientBoosting..........")
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(x_train, y_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_gb.pkl'), 'wb') as gb:
        pickle.dump(clf, gb)

    # AdaBoosting ##
    print("AdaBoosting.........")
    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_ada.pkl'), 'wb') as ada:
        pickle.dump(clf, ada)

    # Naive Bayes ##
    print("Naive Bayes...........")
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    with open(os.path.join(pickle_path, \
        pickle_prefix + '_nb.pkl'), 'wb') as nb:
        pickle.dump(clf, nb)

def classifiers_predict():
    '''
        using classifiers we saved to predict the result
        and to get the accuracy/precision/recall information
        @command example
        python svm_pre.py test_vector_file pickle_prefix
    '''

    testFile = sys.argv[1]
    pickle_prefix = sys.argv[2]

    test_vec = os.path.join(vector_path, testFile)

    with open(test_vec, 'r') as testF:
        test_lines = testF.readlines()

    test_vecs = [line[:-1].split('\t') for line in test_lines]
    x_test = [test_vec[:-1] for test_vec in test_vecs]
    y_test = [test_vec[-1] for test_vec in test_vecs]

    with open(os.path.join(pickle_path, \
        pickle_prefix + '_scaler.pkl'), 'rb') as fs:
        fited_scaler = pickle.load(fs)

    x_test = fited_scaler.transform(x_test)
    
    pickle_subfixs = ['_svm.pkl', '_lr.pkl','_rf.pkl', '_knn.pkl',\
         '_dt.pkl', '_gb.pkl', '_ada.pkl', '_nb.pkl']

    resultFile_path = os.path.join(result_path,\
        pickle_prefix + '_clf_results.txt')
    with open(resultFile_path, 'w') as writeF:
        for pickle_subfix in pickle_subfixs:
            with open(os.path.join(pickle_path, \
                pickle_prefix + pickle_subfix), 'rb') as clfP:
                clf = pickle.load(clfP)

            # things we need to calculate P/R/F1
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            total_cnt = 0
            for idx,vector in enumerate(x_test):
                prediction = clf.predict([vector])
                ground_truth = y_test[idx]
                if prediction == '0' and ground_truth == '0':
                    TN += 1
                elif prediction == '0' and ground_truth == '1':
                    FN += 1
                elif prediction == '1' and ground_truth == '0':
                    FP += 1
                else:
                    TP += 1
                total_cnt += 1
            writeF.write(pickle_subfix + '\n')
            writeF.write('correct_cnt: ' + str(TP + TN) + '\n')
            writeF.write('total_cnt: ' + str(total_cnt) + '\n')
            writeF.write('accuracy: ' + str(float((TP + TN) / total_cnt)) + '\n')
            writeF.write('precision: ' + str(float(TP / (TP + FP))) + '\n')
            writeF.write('recall: ' + str(float(TP/ (TP + FN))) + '\n')
            writeF.write('F1 score:' + str(float((2 * TP) / (2* TP + FP + FN))) + '\n')


if __name__ == "__main__":
    #classifiers_fit()
    classifiers_predict()
