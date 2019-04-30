'''
    @author Qi Sun
    @desc Obtain features of trees similarities
'''

import os
import re
import sys

from nltk.tree import ParentedTree

re_pattern = "[^A-Z()]"


def LCS(str1, str2, str3):

    len1 = len(str1)
    len2 = len(str2)
    len3 = len(str3)

    dp = [[[0 for k in range(len3)] \
        for j in range(len2)] for i in range(len1)]

    for i in range(len1):
        for j in range(len2):
            for k in range(len3):
                if str1[i-1] == str2[j-1] \
                    and str1[i-1] == str3[k-1]:
                    dp[i][j][k] = max(dp[i][j][k], dp[i-1][j-1][k-1]+1)
                else:
                    dp[i][j][k] = max(dp[i-1][j][k], \
                        dp[i][j-1][k], dp[i][j][k-1])

    return dp[len1-1][len2-1][len3-1]


def find_simi(tree_list):

    subtree_cnt = len(tree_list)
    if subtree_cnt < 3:
        return 0

    tree_list = [re.sub(re_pattern, '', tree_str)[:-1]\
                 for tree_str in tree_list]

    lcs_result = 0
    for i in range(subtree_cnt - 2):
        lcs = LCS(tree_list[i], tree_list[i+1], tree_list[i+2])
        lcs = lcs/ max(len(tree_list[i]), \
            len(tree_list[i+1]), len(tree_list[i+2]))
        if lcs > lcs_result:
            lcs_result = lcs

    return lcs_result


def traverse(t):

    subtree_dict = {}

    for subtree in t.subtrees(filter=lambda x: x.label() != 'PU'):
        subtree = re.sub(re_pattern, '', str(subtree))
        if subtree not in subtree_dict:
            subtree_dict[subtree] = 1
        else:
            subtree_dict[subtree] += 1
    
    sorted_subdict = sorted(subtree_dict.items(), key=lambda kv: kv[1])

    vec = [0] * 6
    for subtree in subtree_dict:
        if subtree_dict[subtree] < 7:
            vec[subtree_dict[subtree]-1] += 1
    vec = [float(i)/len(subtree_dict) for i in vec]

    return vec

def main():

    process_path = '../data/processing_data/'
    vec_path = '../data/ft_vec/'

    seg_word = sys.argv[1]
    seg_sent = sys.argv[2]
    writeFile = sys.argv[3]
    seg_word_path = os.path.join(process_path, seg_word)
    seg_sent_path = os.path.join(process_path, seg_sent)
    writePath = os.path.join(vec_path, writeFile)

    with open(seg_word_path, 'r') as swF:
        seg_word_lines = swF.readlines()

    with open(seg_sent_path, 'r') as ssF:
        seg_sent_lines = ssF.readlines()
        
    idx = 0
    tmp_list = []
    with open(writePath, 'w') as writeF:
        for sent in seg_sent_lines:
            if sent == '( (PRN (PU 。)) )\n':
                lcs = find_simi(tmp_list)
                tree = ParentedTree.fromstring(\
                    seg_word_lines[idx][:-1])
                print(lcs)
                vec = traverse(tree)
                #tree.draw()
                vec = [str(lcs)] + [str(v) for v in vec]
                writeF.write('\t'.join(vec) + '\t1\n')

                tmp_list = []
                idx += 1
            else:
                tmp_list.append(sent)


if __name__ == "__main__":
    main()
