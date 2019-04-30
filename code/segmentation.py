'''
	@author Qi Sun
	@desc Segmentation of data with sentence and word levels
'''

import os
import re
import sys
import thulac

liter_path = '../data/literary/'
process_path = '../data/processing_data/'

def seg_sentence():
	
	readFile = sys.argv[2]
	writeFile = sys.argv[3]
	readPath = os.path.join(liter_path, readFile)
	writePath = os.path.join(liter_path, writeFile)

	with open(readPath, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	with open(writePath, 'w', encoding='utf-8') as writeF:
		for line in lines:
			line_list = re.split('，|、|。|；|！', line[:-1])
			if line_list[-1] == "":
				line_list = line_list[:-1]
				
			writeF.write('\n'.join(line_list) + '\n。\n')

def seg_word():

	readFile = sys.argv[2]
	writeFile = sys.argv[3]
	readPath = os.path.join(liter_path, readFile)
	writePath = os.path.join(process_path, writeFile)

	with open(readPath, 'r', encoding='utf-8') as readF:
		lines = readF.readlines()

	with open(writePath, 'w', encoding='utf-8') as writeF:
		thu = thulac.thulac(seg_only = True)
		for line in lines:
			text = thu.cut(line)
			word_list = [word[0] for word in text]
			writeF.write(' '.join(word_list))

def seg_type():

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
