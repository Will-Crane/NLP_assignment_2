#Permutation test
import numpy as np

def Permutation_test(results_A,results_B):
	#Input: 2 maps from filename to classification (filename includes POS or NEG so we can assess if correct classification)
	#Output: p value of Monte Carlo Permutation test with R = 5000

	#if there are files in A or B that the other set of classifications doesn't classify, throw a hissy fit
	if set(results_A.keys()) != set(results_B.keys()):
		raise Exception("input maps to permutation test don't have the same domain")
	file_list = sorted(results_A.keys())
	orig_mean_diff = Mean_dif(results_A,results_B)
	#number of times the mean difference in permuted versions is greater than or equal to the unpermuted mean diff
	s = 0

	for perm_number in range(0,5000):
		#list of 0s and 1s where 0 means swap the results for the file at this given index in file_list
		swap_list = np.random.randint(2, size=len(file_list))
		perm_A = {}
		perm_B = {}
		for file_index in range(0,len(file_list)):
			filename = file_list[file_index]
			if swap_list[file_index]:
				#swap the classification/results:
				perm_A[filename] = results_B[filename]
				perm_B[filename] = results_A[filename]
			else:
				#don't swap
				perm_A[filename] = results_A[filename]
				perm_B[filename] = results_B[filename]
		perm_mean_diff = Mean_dif(perm_A,perm_B)
		if perm_mean_diff >= orig_mean_diff:
			s += 1

	return (s + 1)/5001


def Mean_dif(classifications_A,classifications_B):
	#returns the difference in means between the accuracy of A classifications and B classifications
	correct_A = 0
	for filename in classifications_A:
		if filename[26:29] == classifications_A[filename]:
			correct_A += 1
	mean_A = correct_A/len(classifications_A)

	correct_B = 0
	for filename in classifications_B:
		if filename[26:29] == classifications_B[filename]:
			correct_B += 1
	mean_B = correct_B/len(classifications_B)

	#return the difference in means (abs to force positive)
	return abs(mean_A-mean_B)



