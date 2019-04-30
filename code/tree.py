'''
	@author Qi Sun
	@Desc build tree from NLTK.tree and run tree kernel agrithom
'''

import operator

from nltk.tree import Tree


class TreeNode():
	'''
		class of each node of NLTK tree
	'''
	def __init__(self, num, label):
		super(TreeNode, self).__init__()
		self.num = num
		self.label = label
		self.child_nums = []
		self.child_labels = []
		self.terminal = False
		self.pre_terminal = False

	def add_child_num(self, child_idx):
		self.child_nums.append(child_idx)

	def add_child_label(self, child_label):
		self.child_labels.append(child_label)

	def is_terminal(self):
		self.terminal = True

	def is_pre_terminal(self):
		self.pre_terminal = True


def tree_bfs(tree):
	'''
		traverse a NLTK tree with breath-first method
		and define each tree node with a class
	'''
	tnClassList = []
	nodeList = [tree]
	parentIdx = [-1]

	for idx, curNode in enumerate(nodeList):

		# add child index
		if parentIdx[idx] != -1:
			tnClassList[parentIdx[idx]].add_child_num(idx)

		try:		
			label = curNode.label()
		# if this node is a terminal node(leave)
		except Exception:
			label = curNode
			curTnClass = TreeNode(idx, label)
			curTnClass.is_terminal()
		else:
			curTnClass = TreeNode(idx, label)
			# if this node is a pre-terminal node
			if curNode.height() == 2:
				curTnClass.is_pre_terminal()
			# add children information into nodeList
			for child_idx in range(len(curNode)):
				curChild = curNode[child_idx]
				try:
					child_label = curChild.label()
				except Exception:
					child_label = curChild
				curTnClass.add_child_label(child_label)
				nodeList.append(curChild)
				parentIdx.append(idx)

		# after defining this tree node class, we put it in a list
		tnClassList.append(curTnClass)

	# for tnClass in tnClassList:
	# 	print(tnClass.label)
	# 	print(tnClass.child_nums)
	# 	print(tnClass.child_labels)

	return tnClassList

def cal_treeKernel(tnlist1, tnlist2):
	'''
		using classic tree kernel to calculate number of 
		common tree fragments of two syntactic parsing tree
	'''
	weight = 0.5
	nodeCnt1 = len(tnlist1)
	nodeCnt2 = len(tnlist2)
	dp = [[0 for j in range(nodeCnt2)] for i in range(nodeCnt1)]
	for i in range(nodeCnt1-1, -1, -1):
		for j in range(nodeCnt2-1, -1, -1):
			node1 = tnlist1[i]
			node2 = tnlist2[j]
			# first check whether node1 == node2
			if node1.label != node2.label \
				or operator.eq(node1.child_labels, node2.child_labels) == False:
				continue
			# if node1 == node2 and they are terminal nodes
			if node1.terminal == True and node2.terminal == True:
				dp[i][j] = 1
			# if node1 == node2 and they are pre-terminal nodes
			elif node1.pre_terminal == True and node2.pre_terminal == True:
				dp[i][j] = weight
			# otherwise
			else:
				childCnt = len(node1.child_nums)
				dp[i][j] = weight
				for idx in range(childCnt):
					dp[i][j] *= 1 + dp[node1.child_nums[idx]][node2.child_nums[idx]]
	# tree kernel function is based on counting the
	# number of tree fragments that are common to both parsing trees
	tree_kernel = float(sum(sum(row) for row in dp)/(nodeCnt1+nodeCnt2))
	return tree_kernel

if __name__ == "__main__":
	t1 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased)(NP (D the) (N cat))))")
	t2 = Tree.fromstring("(S (NP (D the)) (VP (V chased)(NP (D the))))")
	#t1 = Tree.fromstring("(S (NP (D the)) (VP (V chased)))")
	#t2 = Tree.fromstring("(S (NP (D the)) (VP (V chased)))")
	tnlist1 = tree_bfs(t1)
	tnlist2 = tree_bfs(t2)
	print(cal_treeKernel(tnlist1, tnlist2))
