'''
	@author Qi Sun
	@desc Segmentation of data with sentence and word levels
	@command example
	python segmentation.py s paral_train_path seg_paral_train_ansi_path
	python segmentation.py w paral_train_path paral_train_ansi_path
'''

import os
import re
import sys
import thulac

dataset_path = '../data/dataset/'
process_path = '../data/processing_data/'
thu = thulac.thulac(seg_only = True)

def seg_sentence():
	'''
		divide a sentence to many clauses
	'''

	clause_re = '，|；|、|,|;|。|！|\.|？|\?|\!'
	readFile = sys.argv[2]
	writeFile = sys.argv[3]
	readPath = os.path.join(dataset_path, readFile)
	writePath = os.path.join(process_path, writeFile)

	with open(readPath, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	with open(writePath, 'w', encoding='utf-8') as writeF:
		for line in lines:
			info = line.split('\t')
			label = info[1][:-1]
			line_list = re.split(clause_re, info[0])
			if line_list[-1] == "":
				line_list = line_list[:-1]
			for line in line_list:
				text = thu.cut(line)
				word_list = [word[0] for word in text]
				writeF.write(' '.join(word_list))
				writeF.write('\n')
			writeF.write(label + '\n')

def seg_word():
	'''
		To segment a sentence with word level
	'''

	readFile = sys.argv[2]
	writeFile = sys.argv[3]
	readPath = os.path.join(dataset_path, readFile)
	writePath = os.path.join(process_path, writeFile)

	with open(readPath, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	with open(writePath, 'w', encoding='utf-8') as writeF:
		
		for line in lines:
			text = thu.cut(line.split('\t')[0])
			word_list = [word[0] for word in text]
			writeF.write(' '.join(word_list))
			writeF.write('\n')

def seg_type():
	'''
		choose which segmentation type you want
	'''

	t = sys.argv[1]
	if t == 's':
		seg_sentence()
	elif t == 'w':
		seg_word()
	else:
		print('Please input collect type of segmentation')
		raise Exception

if __name__ == "__main__":
	seg_type()
