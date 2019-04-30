'''
	@author QI Sun
	@desc get dataset
'''

import os
import re

def main():

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

if __name__ == "__main__":
	main()
