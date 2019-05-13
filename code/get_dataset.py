'''
	@author Qi Sun
	@desc Get dataset and all the processing functions
'''

import os
import re
import random

def collect_neg_samples():
	'''
		collect negative samples from THUCNEWS dataset
	'''

	sent_re = '。|！|\.|？|\?|\!'
	clause_re = '，|；|、|,|;'
	read_path = '../THUCNews'
	write_path = 'thucData.txt'

	with open(write_path, 'w', encoding='utf-8') as writeF:
		for root, dirs, files in os.walk(read_path):
			for file in files:
				filepath = os.path.join(root,file)
				with open(filepath, 'r', encoding='utf-8') as readF:
					for line in readF.readlines():
						sents = re.split(sent_re, line[:-1])
						for sent in sents:
							clauses = re.split(clause_re, sent)
							if len(clauses) > 2:
								writeF.write(sent + '\n')

def filter_standard_paral_samples():
	'''
		simply filter standard parallelism samples 
		we crawled from internet and label each sentence
	'''

	clause_re = '，|；|、|,|;|。|！|\.|？|\?|\!'
	serial_re = r'\d+[、\.，]'
	read_path = '../data/literary/webParallelism2-5.txt'
	write_path = '../data/literary/standard_pos_paral.txt'
	
	with open(read_path, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	with open(write_path, 'w', encoding='utf-8') as writeF:
		for line in lines:
			# we need to remove serial number of each sentence 
			sent = re.sub(serial_re, '', line[:-1])
			clauses = re.split(clause_re, sent)
			# if this sentence didn't reach our basic requirement
			# we won't put this sentence into our dataset
			if len(clauses) > 2 and len(sent) > 30:
				writeF.write(sent + '\t1\n')

def filter_paral_samples():
	'''
		simply filter translated and primary parallelism samples
		after our manuel annotating and save them in a file
	'''

	serial_re = r'\d+[、\.，]'
	read_path = '../data/literary/webParallelism.txt'
	write_path = '../data/literary/annotated_paral.txt'

	with open(read_path, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	with open(write_path, 'w', encoding='utf-8') as writeF:
		for line in lines:
			info = line.split('\t')
			# if this sentence has label and its label is not BAD
			if info[1] == '0\n' or info[1] == '1\n':
				line = re.sub(serial_re, '', line)
				writeF.write(line)

def filter_neg_paral_samples():
	'''
		get negative parallelism samples from THUCNEWS
	'''

	clause_re = '，|；|、|,|;|。|！|\.|？|\?|\!'
	read_path = '../data/processing_data/thucData.txt'
	write_path = '../data/literary/neg_paral.txt'

	with open(read_path, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	with open(write_path, 'w', encoding='utf-8') as writeF:
		negCnt = 0
		cur_idx = 10
		step = 100
		while negCnt < 2000:
			sent = lines[cur_idx].strip()
			clauses = re.split(clause_re, sent)
			if len(clauses) > 2 and len(sent) > 30:
				writeF.write(sent + '\t0\n')
				negCnt += 1
			cur_idx += step
			
def seperate_paral_dataset():
	'''
		we will get our final parallelism dataset
		and then divide it into training set and test set
	'''

	dataset1_path = '../data/literary/standard_pos_paral.txt'
	dataset2_path = '../data/literary/annotated_paral.txt'
	dataset3_path = '../data/literary/neg_paral.txt'
	trainset_path = '../data/dataset/parallelism_train.tsv'
	testset_path = '../data/dataset/parallelsim_test.tsv'

	with open(dataset1_path, 'r', encoding='utf-8') as readF1:
		dataset = readF1.readlines()

	with open(dataset2_path, 'r', encoding='utf-8') as readF2:
		dataset += readF2.readlines()

	with open(dataset3_path, 'r', encoding='utf-8') as readF3:
		dataset += readF3.readlines()
	
	random.shuffle(dataset)

	#print(len(dataset))
	trainCnt = int(len(dataset) * 0.9)
	trainset = dataset[:trainCnt]
	testset = dataset[trainCnt:]

	with open(trainset_path, 'w', encoding='utf-8') as writeF1:
		for train in trainset:
			writeF1.write(train)

	with open(testset_path, 'w', encoding='utf-8') as writeF2:
		for test in testset:
			writeF2.write(test)

def get_dual_annotation():
	'''
		we try to get annotation dual sentences
	'''

	sent_re = '。|！|\.|？|\?|\!'
	clause_re = '，|；|、|,|;'
	readPath = '../data/literary/webDual.txt'
	writePath = '../data/literary/dualAnnoation3.csv'

	with open(readPath, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	with open(writePath, 'w', encoding='utf-8') as writeF:
		for line in lines:
			if line != '\n':
				sents = re.split(sent_re, line[:-1].strip())
				for sent in sents:
					clauses = re.split(clause_re, sent)
					clauseCnt = len(clauses)
					# dual sentence need two clauses 
					# has same number of words
					if clauseCnt % 2 == 0:
						clauseLen1 = 0
						clauseLen2 = 0
						for idx in range(int(clauseCnt/2)):
							clauseLen1 += len(clauses[idx])
						for idx in range(int(clauseCnt/2), clauseCnt):
							clauseLen2 += len(clauses[idx])
						if clauseLen1 == clauseLen2:
							writeF.write(sent + '\n')
						
def seperate_dual_dataset():
	'''
		we will get our final dual dataset
		and then divide it into training set and test set
	'''

	readPath = '../data/literary/dualAnnoation.txt'
	trainset_path = '../data/dataset/dual_train.tsv'
	testset_path = '../data/dataset/dual_test.tsv'

	with open(readPath, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	dataset = [line for line in lines if line.split('\t')[1] != 'B\n']

	random.shuffle(dataset)

	trainCnt = int(len(dataset) * 0.9)
	trainset = dataset[:trainCnt]
	testset = dataset[trainCnt:]

	with open(trainset_path, 'w', encoding='utf-8') as writeF1:
		for train in trainset:
			writeF1.write(train)

	with open(testset_path, 'w', encoding='utf-8') as writeF2:
		for test in testset:
			writeF2.write(test)


if __name__ == "__main__":
	seperate_dual_dataset()
