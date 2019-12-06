#A2-FilePreprocess:
#stores a list of all document texts, preprocessed as described in file_preprocess
#as a pickle file, creates this file and saves to disk if not done already:
import os
import pickle

class Text_data_store:
	#looks after/stores data for both large and original Pang datasets:

	def __init__(self):
		#list of paragraphs in large data store (100,000 docs), each paragraph stored as a list of strings (individual words)
		self.large_data_store = []
		self.format_large_dataset()
		#map from filename (not full path) to list of words in Pang et al 2000 review dataset
		self.pang_data_store = {}
		self.encoded_pang_data_store = {}
		#map from word to integer encoding for that word:
		self.encodings = {}
		self.format_encodings()
		self.format_pang_dataset()
		#Deployment dataset:
		self.deployment_data_store = {}
		self.encoded_deployment_data_store = {}
		self.format_deployment_dataset()

	def format_large_dataset(self):
		if 'pickled_data.pkl' in os.listdir('./'):
			#if the pickled file already exists then load it into the data store
			with open('pickled_data.pkl', 'rb') as f:
				self.large_data_store = pickle.load(f)
		else:
			#must create the pickled list file:
			for foldername in [f for f in os.listdir('assignment_2_large_dataset/') if not f.startswith('.')]:
				for filename in [f for f in os.listdir('assignment_2_large_dataset/' + foldername) if not f.startswith('.')]:
					featurelist = self.file_to_paragraph_list('assignment_2_large_dataset/' + foldername + '/' + filename)
					self.large_data_store.extend(featurelist)
			with open('pickled_data.pkl','wb') as f:
				pickle.dump(self.large_data_store, f, protocol = 2)

	def file_to_paragraph_list(self,filename):
		#input: filename
		#output: list of paragraphs in the file, each paragraph represented by a list of strings (individual words)
		fulltext = ''
		with open(filename, encoding = 'utf8') as f:
			fulltext = f.read()
		#split the file on paragraphs: denoted by "<br /><br />" in the text
		toreturn = []
		for paragraph in fulltext.split("<br /><br />"):
			toreturn.append(paragraph.split())
		return toreturn

	def format_pang_dataset(self):
		#first feature cutoff (freq of 4 or above):
		featuretofreq = {}
		for foldername in [f for f in os.listdir('assignment_2_pang_dataset/') if not f.startswith('.')]:
			for filename in [f for f in os.listdir('assignment_2_pang_dataset/' + foldername) if not f.startswith('.')]:
				with open('assignment_2_pang_dataset/' + foldername + "/" + filename, encoding = 'utf8') as f:
					for word in f.read().split():
						encoding = self.encodings[word]
						if encoding in featuretofreq:
							featuretofreq[encoding] += 1
						else:
							featuretofreq[encoding] = 0
		included_encodings = []
		for encoding in featuretofreq.keys():
			if featuretofreq[encoding] >= 4:
				included_encodings.append(encoding)

		for foldername in [f for f in os.listdir('assignment_2_pang_dataset/') if not f.startswith('.')]:
			for filename in [f for f in os.listdir('assignment_2_pang_dataset/' + foldername) if not f.startswith('.')]:
				full_filename = ('assignment_2_pang_dataset/' + foldername + '/' + filename)
				with open(full_filename, encoding = 'utf8') as f:
					fulltext = f.read()
					encoded_features = []
					features = []
					for word in fulltext.split():
						if self.encodings[word] in included_encodings:
							encoded_features.append(self.encodings[word])
							features.append(word)
					self.pang_data_store[full_filename] = features
					self.encoded_pang_data_store[full_filename] = encoded_features

	def file_to_wordlist(self,filename):
		if filename in self.pang_data_store:
			return self.pang_data_store[filename]
		elif filename in self.deployment_data_store:
			return self.deployment_data_store[filename]
		else:
			raise Exception("filename {0} not found in pang_data_store or deployment_data_store".format(filename))

	def filelist_to_wordlist_dict(self,filelist):
		toreturn = {}
		for filename in filelist:
			toreturn[filename] = self.file_to_wordlist(filename)
		return toreturn

	def encoded_feature_dict(self,filelist):
		#returns the dict from filename to list of encoded features in file
		toreturn = {}
		for filename in filelist:
			if filename in self.encoded_pang_data_store:
				toreturn[filename] = self.encoded_pang_data_store[filename]
			elif filename in self.encoded_deployment_data_store:
				toreturn[filename] = self.encoded_deployment_data_store[filename]
			else:
				raise Exception("filename {0} not found in encoded_pang_data_store or encoded_dply_data_store".format(filename))
		return toreturn

	def format_deployment_dataset(self):
		#feature cutoff:
		featuretofreq = {}
		for foldername in ['assignment_2_dply_dataset/POS','assignment_2_dply_dataset/NEG']:
			for filename in [f for f in os.listdir(foldername) if not f.startswith('.')]:
				with open(foldername + "/" + filename, encoding = 'utf8') as f:
					for word in f.read().split():
						encoding = self.encodings[word]
						if encoding in featuretofreq:
							featuretofreq[encoding] += 1
						else:
							featuretofreq[encoding] = 0
		included_encodings = []
		for encoding in featuretofreq.keys():
			if featuretofreq[encoding] >= 4:
				included_encodings.append(encoding)

		for foldername in ['assignment_2_dply_dataset/POS','assignment_2_dply_dataset/NEG']:
			for filename in [f for f in os.listdir(foldername) if not f.startswith('.')]:
				full_filename = (foldername + '/' + filename)
				with open(full_filename, encoding = 'utf8') as f:
					fulltext = f.read()
					encoded_features = []
					features = []
					for word in fulltext.split():
						if self.encodings[word] in included_encodings:
							encoded_features.append(self.encodings[word])
							features.append(word)
					self.deployment_data_store[full_filename] = features
					self.encoded_deployment_data_store[full_filename] = encoded_features

	def format_encodings(self):
		allwords = set()
		for foldername in [f for f in os.listdir('assignment_2_pang_dataset/') if not f.startswith('.')]:
			for filename in [f for f in os.listdir('assignment_2_pang_dataset/' + foldername) if not f.startswith('.')]:
				full_filename = ('assignment_2_pang_dataset/' + foldername + '/' + filename)
				with open(full_filename, encoding = 'utf8') as f:
					fulltext = f.read()
					allwords.update(fulltext.split())
		for foldername in ['assignment_2_dply_dataset/POS','assignment_2_dply_dataset/NEG']:
			for filename in [f for f in os.listdir(foldername) if not f.startswith('.')]:
				full_filename = (foldername + '/' + filename)
				with open(full_filename, encoding = 'utf8') as f:
					fulltext = f.read()
					allwords.update(fulltext.split())
		encodingnumber = 1
		for word in allwords:
			self.encodings[word] = encodingnumber
			encodingnumber += 1

