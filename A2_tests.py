#Assignment 2 tests:
from A2_SVM_classifier import SVM_classifier
from A2_Text_data_store import Text_data_store
from A2_Permutation_test import Permutation_test
from NaiveBayes import NaiveBayes
import logging
import math
import os
logging.basicConfig(filename="test.log", level=logging.DEBUG)

class A2_tests:

	def __init__(self,paramlist):
		#list of filenames in validation corpus
		self.validationcorpus = []
		#list of filenames in trainingcorpus
		self.trainingcorpus = []
		self.createvalandtrain()
		#svm class object:
		self.svm = SVM_classifier()
		self.nb = NaiveBayes()
		logging.basicConfig(filename="test.log", level=logging.DEBUG)
		#uncomment whichever test you want to run N.B. tweaking takes a long time
		#self.tweak()
		#self.tweaking_significant()
		#self.NB_vs_SVM()
		#self.NB_vs_SVM_deploy()
		#self.case_studies()

	def tweak(self):
		results_map = {}
		count = 0
		for vsize in [50,75,100,150]:
			for window in [2,4,6,8]:
				for min_count in [1,2,3,4]:
					for dm in [0,1]:
						count += 1
						print("Progress: " + str(count) + "/128")
						print("Params: " + str(vsize) + ", " + str(window) + ", " + str(min_count) + ", " + str(dm))
						self.svm.retrain_doc2vec(vsize, window, min_count, 4, dm)
						classification_map = self.svm.evaluate(self.trainingcorpus,self.validationcorpus)
						accuracy = self.get_accuracy(classification_map)
						logging.info("\n\n" + "Hey, hey, over here, look at these:" + str(vsize) + ", " + str(window) + ", " + str(min_count) + ", " + str(dm) + ": " + str(accuracy))
						results_map[(vsize, window, min_count, dm)] = accuracy
		for key in results_map.keys():
			(v, w, m, d) = key
			print(str(v) + ", " + str(w) + ", " + str(m) + ", " + str(d) + ": " + str(results_map[key]))

	def tweaking_significant(self):
		#using 10-fold CV use doc2vec to get a list of classifications
		d2vclassificationsbest = {}
		self.svm.retrain_doc2vec(50, 2, 1, 1500, 0)
		for k in range(0,10):
			(train,test) = self.tenfoldCV(self.trainingcorpus,k)
			classification_map = self.svm.evaluate(train,test)
			d2vclassificationsbest.update(classification_map)
			print("D2V:  fold setup: " + str(k) + ",   accuracy: " + str(self.get_accuracy(classification_map)))
		print("===============================================")

		#using 10-fold CV use doc2vec to get a list of classifications
		d2vclassificationsworst = {}
		self.svm.retrain_doc2vec(50, 4, 1, 1500, 1)
		for k in range(0,10):
			(train,test) = self.tenfoldCV(self.trainingcorpus,k)
			classification_map = self.svm.evaluate(train,test)
			d2vclassificationsworst.update(classification_map)
			print("D2V:  fold setup: " + str(k) + ",   accuracy: " + str(self.get_accuracy(classification_map)))
		print("===============================================")

		p = Permutation_test(d2vclassificationsbest,d2vclassificationsworst)
		print("Perm test: " + str(p))

	def NB_vs_SVM(self):
		nbclassifications = {}
		print("NB:")
		for k in range(0,10):
			(train,test) = self.tenfoldCV(self.trainingcorpus,k)
			classification_map = self.nb.Unigramsfrequency(train,test)
			nbclassifications.update(classification_map)
			print(str(self.get_accuracy(classification_map)))
		print(str(self.get_accuracy(nbclassifications)))
		print("===============================================")

		#using 10-fold CV use doc2vec to get a list of classifications
		d2vclassificationsbest = {}
		self.svm.retrain_doc2vec(50, 2, 1, 1500, 0)
		for k in range(0,10):
			(train,test) = self.tenfoldCV(self.trainingcorpus,k)
			classification_map = self.svm.evaluate(train,test)
			d2vclassificationsbest.update(classification_map)
			print("D2V:  fold setup: " + str(k) + ",   accuracy: " + str(self.get_accuracy(classification_map)))
		print("===============================================")

		#using 10-fold CV use assignment 1 SVM to get a list of classifications
		ass1classifications = {}
		for k in range(0,10):
			(train,test) = self.tenfoldCV(self.trainingcorpus,k)
			classification_map = self.svm.ass_1_SVM(train,test)
			ass1classifications.update(classification_map)
			print("Ass 1: fold setup: " + str(k) + ",   accuracy: " + str(self.get_accuracy(classification_map)))
		print("===============================================")

		print("NB acc: " + str(self.get_accuracy(nbclassifications)))
		print("SVM1 (old one) acc: " + str(self.get_accuracy(ass1classifications)))
		print("SVM2 (D2V) acc: " + str(self.get_accuracy(d2vclassificationsbest)))

		p1 = Permutation_test(nbclassifications,ass1classifications)
		p2 = Permutation_test(nbclassifications,d2vclassificationsbest)
		p3 = Permutation_test(ass1classifications,d2vclassificationsbest)

		print("NB vs SVM1: " + str(p1))
		print("NB vs SVM2: " + str(p2))
		print("SVM1 vs SVM2: " + str(p3))

	def NB_vs_SVM_deploy(self):
		deployment_files = [('assignment_2_dply_dataset/NEG/' + f) for f in os.listdir('assignment_2_dply_dataset/NEG/') if not f.startswith('.')]
		deployment_files.extend(['assignment_2_dply_dataset/POS/' + f for f in os.listdir('assignment_2_dply_dataset/POS/') if not f.startswith('.')])
		nbclassifications = self.nb.Unigramsfrequency(self.trainingcorpus,deployment_files)

		#using 10-fold CV use doc2vec to get a list of classifications
		self.svm.retrain_doc2vec(50, 2, 1, 1500, 0)
		d2vclassificationsbest = self.svm.evaluate(self.trainingcorpus,deployment_files)

		#using 10-fold CV use assignment 1 SVM to get a list of classifications
		ass1classifications = self.svm.ass_1_SVM(self.trainingcorpus,deployment_files)

		print("NB acc: " + str(self.get_accuracy(nbclassifications)))
		print("SVM1 (old one) acc: " + str(self.get_accuracy(ass1classifications)))
		print("SVM2 (D2V) acc: " + str(self.get_accuracy(d2vclassificationsbest)))

		p1 = Permutation_test(nbclassifications,ass1classifications)
		p2 = Permutation_test(nbclassifications,d2vclassificationsbest)
		p3 = Permutation_test(ass1classifications,d2vclassificationsbest)

		print("NB vs SVM1: " + str(p1))
		print("NB vs SVM2: " + str(p2))
		print("SVM1 vs SVM2: " + str(p3))

	def case_studies(self):
		deployment_files = [('assignment_2_dply_dataset/NEG/' + f) for f in os.listdir('assignment_2_dply_dataset/NEG/') if not f.startswith('.')]
		deployment_files.extend(['assignment_2_dply_dataset/POS/' + f for f in os.listdir('assignment_2_dply_dataset/POS/') if not f.startswith('.')])
		nbclassifications = self.nb.Unigramsfrequency(self.trainingcorpus,deployment_files)

		self.svm.retrain_doc2vec(50, 2, 1, 1500, 0)
		d2vclassificationsbest = self.svm.evaluate(self.trainingcorpus,deployment_files)

		ass1classifications = self.svm.ass_1_SVM(self.trainingcorpus,deployment_files)

		print("NB acc: " + str(self.get_accuracy(nbclassifications)))
		print("SVM1 (old one) acc: " + str(self.get_accuracy(ass1classifications)))
		print("SVM2 (D2V) acc: " + str(self.get_accuracy(d2vclassificationsbest)))

		print("===========================================")

		for filename in ass1classifications.keys():
			filetype = filename[26:29]
			if ass1classifications[filename] != filetype and nbclassifications[filename] != filetype and d2vclassificationsbest[filename] == filetype:
				print(filename)

		pos = 0
		neg = 0
		for filename in ass1classifications.keys():
			filetype = filename[26:29]
			if filetype != d2vclassificationsbest[filename]:
				if filetype == 'POS':
					pos += 1
				if filetype == 'NEG':
					neg += 1
		print("POS: " + str(pos))
		print("NEG: " + str(neg))

			

	def createvalandtrain(self):
		index = 0
		pang_files = [('assignment_2_pang_dataset/NEG/' + f) for f in os.listdir('assignment_2_pang_dataset/NEG/')]
		pang_files.extend(['assignment_2_pang_dataset/POS/' + f for f in os.listdir('assignment_2_pang_dataset/POS/')])
		for filename in pang_files:
			if(index < 9):
				self.trainingcorpus.append(filename)
				index += 1
			else:
				self.validationcorpus.append(filename)
				index = 0
		#double check no validationcorpus files are in trainingcorpus:
		for filename in self.validationcorpus:
			if filename in self.trainingcorpus:
				raise Exception("File found in both testing and training data: {0}".format(filename))
"""
	def CV_tests(self,params):
		#using 10-fold CV use assignment 1 SVM to get a list of classifications
		ass1classifications = {}
		for k in range(0,10):
			(train,test) = self.tenfoldCV(self.trainingcorpus,k)
			classification_map = self.svm.ass_1_SVM(train,test)
			ass1classifications.update(classification_map)
			print("Ass 1: fold setup: " + str(k) + ",   accuracy: " + str(self.get_accuracy(classification_map)))
		print("===============================================")

		#using 10-fold CV use doc2vec to get a list of classifications
		d2vclassifications = {}
		(vsize, wndw, mcount, wrk,dm) = params
		#self.svm.retrain_doc2vec(vsize, wndw, mcount, wrk)
		for k in range(0,10):
			(train,test) = self.tenfoldCV(self.trainingcorpus,k)
			classification_map = self.svm.evaluate(train,test)
			d2vclassifications.update(classification_map)
			print("D2V:  fold setup: " + str(k) + ",   accuracy: " + str(self.get_accuracy(classification_map)))
		print("===============================================")
		
		#perform the permutation test on these results
		p = Permutation_test(d2vclassifications,ass1classifications)
		print("Perm test p val:" + str(p))
		print("ass1 acc: " + str(self.get_accuracy(ass1classifications)))
		print("d2v acc: " + str(self.get_accuracy(d2vclassifications)))
		return 1

	def Deployment_test(self,params):
		deployment_files = [('assignment_2_dply_dataset/NEG/' + f) for f in os.listdir('assignment_2_dply_dataset/NEG/') if not f.startswith('.')]
		deployment_files.extend(['assignment_2_dply_dataset/POS/' + f for f in os.listdir('assignment_2_dply_dataset/POS/') if not f.startswith('.')])
		#using 10-fold CV use assignment 1 SVM to get a list of classifications
		ass1classifications = {}
		for k in range(0,10):
			(train,test) = self.tenfoldCV(deployment_files,k)
			classification_map = self.svm.ass_1_SVM(train,test)
			ass1classifications.update(classification_map)
			print("Ass 1: fold setup: " + str(k) + ",   accuracy: " + str(self.get_accuracy(classification_map)))
		print("===============================================")

		#using 10-fold CV use doc2vec to get a list of classifications
		d2vclassifications = {}
		(vsize, wndw, mcount, wrk, dm) = params
		#self.svm.retrain_doc2vec(vsize, wndw, mcount, wrk)
		for k in range(0,10):
			(train,test) = self.tenfoldCV(deployment_files,k)
			classification_map = self.svm.evaluate(train,test)
			d2vclassifications.update(classification_map)
			print("D2V:  fold setup: " + str(k) + ",   accuracy: " + str(self.get_accuracy(classification_map)))
		print("===============================================")
		
		#perform the permutation test on these results
		p = Permutation_test(d2vclassifications,ass1classifications)
		print("Perm test p val:" + str(p))
		print("ass1 acc: " + str(self.get_accuracy(ass1classifications)))
		print("d2v acc: " + str(self.get_accuracy(d2vclassifications)))
		return 1
"""

	def tenfoldCV(self,filelist,k):
		#returns (train,test) using 10 fold round-robin CV (k used to select different fold as train/test)
		if k<0 or k>9:
			raise Exception("invalid k value for tenfoldCV {0}".format(str(k)))
		index = k
		train = []
		test = []
		for filename in filelist:
			if k == 9:
				test.append(filename)
				k = 0
			else:
				train.append(filename)
				k += 1
		return (train, test)

	def get_accuracy(self,classification_map):
		correct = 0
		total = 0
		for filename in classification_map:
			if filename[26:29] == classification_map[filename]:
				correct += 1
			total += 1
		return correct/total

b = A2_tests(paramlist)