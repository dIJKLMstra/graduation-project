import random
from sklearn import svm

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

if __name__ == "__main__":
	svm_predict()
