#training the doc2vec model:
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

class Doc_2_Vec:

	def __init__(self, data_store):
		self.documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data_store)]
		self.model = None
		self.loadmodel()

	def trainmodel(self, vsize, wndw, mcount, wrk, dmv):
		print("Training model: this may take a while...")
		self.model = Doc2Vec(self.documents, vector_size=vsize, window=wndw, workers=4, min_count=mcount, dm=dmv, seed=1, epochs = 10)
		model_filename = get_tmpfile("D2V_model")
		self.model.save(model_filename)

	def loadmodel(self):
		#if there is a model as a temp file, load it in as our model, else train a new model with default settings:
		try:
			self.model = Doc2Vec.load(get_tmpfile("D2V_model"))
		except:
			self.trainmodel(50,2,1,4,0)

	def wordlisttovector(self,wordlist):
		return self.model.infer_vector(wordlist)