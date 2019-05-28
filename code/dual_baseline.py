'''
    @author Qi Sun
    @desc using classifiers to predict the results
'''

import os
import sys

vector_path = '../data/ft_vec'
pickle_path = '../data/clf_pickles'
result_path = '../data/result'

def pos_simi(pos1, pos2):
    '''
        calculate the similarity of two pos tagging
    '''

    poslen = len(pos1)
    sameCnt = 0
    for i in range(poslen):
        if pos1[i] == pos2[i]:
            sameCnt += 1
    return float(sameCnt/poslen)

def edit(pos1, pos2):
    '''
        edit distance of two pos tagging
    '''

    poslen = len(pos1)
    matrix = [[i+j for j in range(poslen + 1)] for i in range(poslen + 1)]
 
    for i in range(1, poslen + 1):
        for j in range(1, poslen + 1):
            if pos1[i-1] == pos2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1,\
                matrix[i][j-1]+1, matrix[i-1][j-1]+d)
 
    return float(matrix[poslen][poslen]/poslen)

def get_features():
    '''
        get the features of dual pos taggings
    '''

    trainPath = '../data/processing_data/dual_train_POS.txt'
    testPath = '../data/processing_data/dual_test_POS.txt'
    trainVecPath = os.path.join(vector_path, 'dual_train_vec.tsv')
    testVecPath = os.path.join(vector_path, 'dual_test_vec.tsv')

    with open(trainPath, 'r') as trainF:
        trainLines = trainF.readlines()

    with open(testPath, 'r') as testF:
        testLines = testF.readlines()

    with open(trainVecPath, 'w') as vecTrain:
        for line in trainLines:
            info = line.split('\t')
            if len(info[0]) != len(info[1]):
                continue
            vecTrain.write(str(pos_simi(info[0], info[1])) +\
                           '\t' + str(edit(info[0], info[1])) + '\t' + info[-1])

    with open(testVecPath, 'w') as vecTest:
        for line in testLines:
            info = line.split('\t')
            if len(info[0]) != len(info[1]):
                continue
            vecTest.write(str(pos_simi(info[0], info[1])) +\
                           '\t' + str(edit(info[0], info[1])) + '\t' + info[-1])

