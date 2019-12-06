#A2-SVM_classifier
import subprocess
import os
import math
from A2_Text_data_store import Text_data_store
from A2_doc2vec_methods import Doc_2_Vec

class SVM_classifier:

	def __init__(self):
		self.tds = Text_data_store()
		self.d2v = Doc_2_Vec(self.tds.large_data_store)

	def evaluate(self, training_files, testing_files):
		#using the given files, trains and evaluates an SVM, returning the map from filename to classification outputted by the model:
		trainingvectormap = self.createvectors(self.tds.filelist_to_wordlist_dict(training_files))
		testingvectormap = self.createvectors(self.tds.filelist_to_wordlist_dict(sorted(testing_files)))
		self.Formatdata(trainingvectormap,testingvectormap)
		print('Training SVM...')
		#train the model:
		subprocess.call("./svm_learn SVMtrain.dat SVMmodel.dat" , shell=True)
		print('Classifying test reviews')
		#classify the test files:
		subprocess.call("./svm_classify SVMtest.dat SVMmodel.dat SVMoutput.dat" , shell=True)
		#return the classifications:
		return self.Classificationlist(testing_files)

	def retrain_doc2vec(self, vsize, wndw, mcount, wrk, dm):
		self.d2v.trainmodel(vsize, wndw, mcount, wrk, dm)

	def Formatdata(self, trainingvectormap, testingvectormap):
		#trainingvectormap = map from filename to vector for that file
		#testngvectormap = map from filename to vector for that file
		#format the data into the files: "SVMtrain.dat" and "SVMtest.dat"

		#first remove any old files:
		if "SVMtrain.dat" in os.listdir('.'): os.remove('SVMtrain.dat')
		if "SVMtest.dat" in os.listdir('.'): os.remove('SVMtest.dat')
		if "SVMmodel.dat" in os.listdir('.'): os.remove('SVMmodel.dat')
		if "SVMoutput.dat" in os.listdir('.'): os.remove('SVMoutput.dat')

		#training file:
		training_file = open("SVMtrain.dat","w+")
		for filename in trainingvectormap.keys():
			trainingvectors = trainingvectormap[filename]
			#the string of data that will be written as a single line for this file
			datastring = ''
			filetype = filename[26:29] #either 'POS' or 'NEG'
			if filetype == 'POS': datastring = datastring + '1 '
			elif filetype == 'NEG': datastring = datastring + '-1 '
			else: raise Exception("invalid filename: {0}".format(filename))
			for i in range(len(trainingvectors)):
				datastring = datastring + str(i+1) + ':' + str(trainingvectors[i]) + ' '
			datastring = datastring + "#" + filename #this is only for debugging purposes
			#write the datastring to the file:
			training_file.write(datastring)
			training_file.write("\n")
		training_file.close()

		#testing file:
		testing_file = open("SVMtest.dat","w+")
		for filename in sorted(testingvectormap.keys()):
			testingvectors = testingvectormap[filename]
			#the string of data that will be written as a single line for this file
			datastring = ''
			filetype = filename[26:29] #either 'POS' or 'NEG'
			if filetype == 'POS': datastring = datastring + '1 '
			elif filetype == 'NEG': datastring = datastring + '-1 '
			else: raise Exception("invalid filename: {0}".format(filename))
			for i in range(len(testingvectors)):
				datastring = datastring + str(i+1) + ':' + str(testingvectors[i]) + ' '
			datastring = datastring + "#" + filename #this is only for debugging purposes

			#write the datastring to the file:
			testing_file.write(datastring)
			testing_file.write("\n")
		testing_file.close()

	def createvectors(self,mapfilenametolistofwords):
		#take in map from filename to list of words in file ()
		#create map from filename to file vector:
		vectormap = {}
		for filename in mapfilenametolistofwords:
			vectormap[filename] = self.d2v.wordlisttovector(mapfilenametolistofwords[filename])
		return vectormap

	def Classificationlist(self, testing_files):
		#given the list of testing_files
		#return the map from filename to classification given by model (by reading the output file)
		classificationmap = {}
		output_file = open("SVMoutput.dat")
		#we use the fact that the testingfiles were sorted before being formatted into SVMtest.dat
		for filename in sorted(testing_files):
			predictedsentiment = 'NEG'
			if float(output_file.readline()) > 0:
				predictedsentiment = 'POS'
			classificationmap[filename] = predictedsentiment
		return classificationmap

	def ass_1_SVM(self, training_files, testing_files):
		#Unigrams frequency:
		training_feat_dict = self.tds.encoded_feature_dict(training_files)
		testing_feat_dict = self.tds.encoded_feature_dict(testing_files)

		#format: map from filename to normalised doc vector: ordered list of (featurenum, value) pairs
		#==========================================
		trainingmap = {}
		testingmap = {}
		#trainingfeatures:
		for filename in training_feat_dict.keys():
			featuretofrequency = {}
			for feature in training_feat_dict[filename]:
				if feature in featuretofrequency.keys():
					featuretofrequency[feature] += 1
				else:
					featuretofrequency[feature] = 1
			trainingmap[filename] = self.Normaliseandsort(featuretofrequency)
		#testingfeatures:
		for filename in testing_feat_dict.keys():
			featuretofrequency = {}
			for feature in testing_feat_dict[filename]:
				if feature in featuretofrequency.keys():
					featuretofrequency[feature] += 1
				else:
					featuretofrequency[feature] = 1
			testingmap[filename] = self.Normaliseandsort(featuretofrequency)

		#Format the data into the train,test files:
		#==========================================
		if "SVMtrain.dat" in os.listdir('.'): os.remove('SVMtrain.dat')
		if "SVMtest.dat" in os.listdir('.'): os.remove('SVMtest.dat')
		if "SVMmodel.dat" in os.listdir('.'): os.remove('SVMmodel.dat')
		if "SVMoutput.dat" in os.listdir('.'): os.remove('SVMoutput.dat')

		#training file:
		training_file = open("SVMtrain.dat","w+")
		for filename in trainingmap.keys():
			#the string of data that will be written as a single line for this file
			datastring = ''
			filetype = filename[26:29] #either 'POS' or 'NEG'
			if filetype == 'POS': datastring = datastring + '1 '
			elif filetype == 'NEG': datastring = datastring + '-1 '
			else: raise Exception("invalid filename: {0}".format(filename))

			for (feature, value) in trainingmap[filename]:
				datastring = datastring + str(feature) + ':' + str(value) + ' '
			datastring = datastring + "#" + filename #this is only for debugging purposes

			#write the datastring to the file:
			training_file.write(datastring)
			training_file.write("\n")
		training_file.close()

		#testing file:
		testing_file = open("SVMtest.dat","w+")
		for filename in testingmap.keys():
			#the string of data that will be written as a single line for this file
			datastring = ''
			filetype = filename[26:29] #either 'POS' or 'NEG'
			if filetype == 'POS': datastring = datastring + '1 '
			elif filetype == 'NEG': datastring = datastring + '-1 '
			else: raise Exception("invalid filename: {0}".format(filename))

			for (feature, value) in testingmap[filename]:
				datastring = datastring + str(feature) + ':' + str(value) + ' '
			datastring = datastring + "#" + filename #this is only for debugging purposes

			#write the datastring to the file:
			testing_file.write(datastring)
			testing_file.write("\n")
		testing_file.close()

		#Train the SVM model and run the classifier:
		#==========================================
		print('Training SVM...')
		#train the model:
		subprocess.call("./svm_learn SVMtrain.dat SVMmodel.dat" , shell=True)
		print('Classifying test reviews')
		#classify the test files:
		subprocess.call("./svm_classify SVMtest.dat SVMmodel.dat SVMoutput.dat" , shell=True)
		#return the classifications:
		return self.Classificationlist(testing_files)

	def Normaliseandsort(self, fvec):
		#takes in a dictionary from feature to frequency and returns the sorted list of:
		#(featurenum, normalised frequency value)
		total = 0
		normfeatvec = []
		for value in fvec.values():
			total += value * value
		veclen = math.sqrt(total)
		for featurenum in sorted(fvec.keys()):
			normfeatvec.append((featurenum,(fvec[featurenum]/veclen)))
		return normfeatvec