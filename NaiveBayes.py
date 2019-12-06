#Naive Bayes implementation
from A2_Text_data_store import Text_data_store
import math

class NaiveBayes:
	def __init__(self):
		self.tds = Text_data_store()
		self.smoothing = True

	def Unigramsfrequency(self, trainingdata, testingdata):
		#gets a map from filename to encoded unigram features for the training set
		modeltrainingfeatures = self.tds.filelist_to_wordlist_dict(trainingdata)
		#train the model on these training features
		model = self.Trainmodel(modeltrainingfeatures)
		#gets a map from filename to encoded unigram features for the testing set
		modeltestingfeatures = self.tds.filelist_to_wordlist_dict(testingdata)
		#classify the sentiment of the files in the testing set
		classifications = self.Testmodel(model, modeltestingfeatures)
		#return this map from filename to classification
		return classifications

	def Trainmodel(self, trainingfeatures):
		#trainingfeatures is a map from filename to a list of the trainingfeatures in the file
		#map from features to (positive log prob, negative log prob) pair
		tmodel = {}

		#map from feature to number of occurences in POS files
		featureposfreq = {}
		totalposfeatures = 0
		#map from feature to number of occurences in NEG files
		featurenegfreq = {}
		totalnegfeatures = 0

		#a set containing all the features seen across the training data set:
		allfeatures = set([])

		#count up feature frequencies acorss training set, creating the above data structures
		for filename in trainingfeatures.keys():
			filetype = filename[26:29] #either 'POS' or 'NEG'
			if(filetype == 'POS'):
				for feature in trainingfeatures[filename]:
					if feature in featureposfreq:
						featureposfreq[feature] += 1
					else:
						featureposfreq[feature] = 1
					totalposfeatures += 1
					allfeatures.add(feature)
			elif(filetype == 'NEG'):
				for feature in trainingfeatures[filename]:
					if feature in featurenegfreq:
						featurenegfreq[feature] += 1
					else:
						featurenegfreq[feature] = 1
					totalnegfeatures += 1
					allfeatures.add(feature)
			else: raise Exception('illegal filename when training NB model: {0}'.format(filename))

		numuniquefeatures = len(allfeatures)
		for feature in allfeatures:
			featurepos = -float('inf') #Log 0
			featureneg = -float('inf') #Log 0
			if self.smoothing:
				featurepos = math.log(1.0/(totalposfeatures + self.smoothing*numuniquefeatures))
				featureneg = math.log(1.0/(totalnegfeatures + self.smoothing*numuniquefeatures))
			if feature in featureposfreq.keys():
				featurepos = math.log(featureposfreq[feature] * 1.0/(totalposfeatures + self.smoothing*numuniquefeatures))
			if feature in featurenegfreq.keys():
				featureneg = math.log(featurenegfreq[feature] * 1.0/(totalnegfeatures + self.smoothing*numuniquefeatures))
			tmodel[feature] = (featurepos, featureneg)
		return tmodel

	def Testmodel(self, model, testingfeatures):
		#map from filename in testingfeatures map to it's classification ('POS' or 'NEG')
		classifications = {}

		for filename in testingfeatures.keys():
			poslogsum = 0
			neglogsum = 0
			for feature in testingfeatures[filename]:
				if feature in model.keys():
					(plp,nlp) = model[feature]
					poslogsum += plp
					neglogsum += nlp
			if (poslogsum >= neglogsum):
				classifications[filename] = 'POS'
			else:
				classifications[filename] = 'NEG'

		return classifications

	def Getaccuracy(self, classificationlist):
		#takes in a map from filename to classification and returns the accuracy of the classifications:
		totalcorrect = 0
		totalclassified = 0
		for filename in classificationlist:
			actualsent = filename[:3]
			if(classificationlist[filename] == actualsent):
				totalcorrect += 1
			totalclassified += 1
		return (totalcorrect/totalclassified)