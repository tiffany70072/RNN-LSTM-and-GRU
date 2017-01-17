import pickle
import string # punctuation
import numpy as np
from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from keras.layers.recurrent import SimpleRNN, LSTM, GRU

class Model(object):
	def __init__(self):
		self.method = 1
		self.numSub = 7
		self.length = 15 # number of words in the longest title 
		self.numTrain = 10000 
		self.dimWordVector = 300
		self.methodName = ["dnn", "rnn", "lstm", "gru"]

		self.batch = 500
		self.epoch = 50
		
	def load(self):
		self.title = pickle.load(open("data/mix-title", "rb"))
		self.sub = pickle.load(open("data/mix-sub", "rb"))
		self.doc = pickle.load(open("data/mix-content", "rb"))
		print "Num = ", len(self.title), len(self.sub)

	def save(self):
		pickle.dump(self.title, open("data/mix-title", "wb"))
		pickle.dump(self.sub, open("data/mix-sub", "wb"))

	def wordVector(self, run_again):
		print "Word to Vector"
		if run_again == 1:
			self.doc = self.title + self.doc
			print "Num = ", len(self.doc)
			self.wordModel = Word2Vec(self.doc, size = self.dimWordVector, window = 5, min_count = 5, workers = 2, iter = 30)
			#self.wordModel.init_sims(replace = True)
			self.wordModel.save("model-wordVector")
		else:
			self.wordModel = Word2Vec.load("model-wordVector")  # you can continue training with the loaded model!
		
	def aveEmbedding(self, run_again):
		print "Word Embedding"
		if run_again == 1:
			self.x = np.zeros([len(self.sub), self.dimWordVector])
			noVector = []
			for i in range(len(self.sub)):
				temp = np.zeros([self.dimWordVector])
				count = 0.0
				for j in range(len(self.title[i])):
					try:
						temp += self.wordModel[self.title[i][j]]
						count += 1.0
					except KeyError: continue

				if count != 0: self.x[i] = np.array(temp/count)
				else: # if no word of the title in the training data 
					noVector.append(i)
					#print i,
			print "\nNum of no word vector = ", len(noVector)
			#pickle.dump(self.titleVector, open("model-titleVector62", "wb"))
		else:
			self.x = pickle.load(open("model-titleVector", "rb"))
		print "x = ", self.x.shape

	def embedding(self):
		run_again = 1
		print "Word Embedding"
		a = len(self.sub)
		b = self.length
		c = self.dimWordVector

		if run_again == 1:
			self.x = np.zeros([a, b, c])
			noVector = []
			for i in range(a):
				for j in range(b):
					if j < len(self.title[i]):
						try: self.x[i][j] = self.wordModel[self.title[i][j]]
						except KeyError: self.x[i][j] = np.zeros([self.dimWordVector])
						
					else: self.x[i][j] = np.zeros([self.dimWordVector])
			#pickle.dump(self.titleVector, open("model-titleVector62", "wb"))
		else:
			self.titleVector = pickle.load(open("model-titleVector", "rb"))
		print "x = ", self.x.shape, self.x[0][0][0]

	def setXY(self):
		self.y = np.zeros([len(self.sub), self.numSub])
		for i in range(len(self.sub)): self.y[i][self.sub[i]-1] = 1
		
		self.y_test = self.y[self.numTrain:]
		self.y = self.y[:self.numTrain]
		self.x_test = self.x[self.numTrain:]
		self.x = self.x[:self.numTrain]

		print "x = ", self.x.shape, ", y = ", self.y.shape
		print "x_test = ", self.x_test.shape, ", y_test = ", self.y_test.shape
		#print self.x_test[:30][0]

	def output(self, result):
		print result.shape
		fout = open("result-" + self.methodName[self.method] + ".csv", 'w')
		#self.ans = np.zeros(result.shape[0])
		#fout.write("ID,class\n")
		for i in range(result.shape[0]):
			maxP = 0.0
			maxId = -1
			for j in range(self.numSub):
				if result[i][j] > maxP:
					maxP = result[i][j]
					maxId = j
			#self.ans[i] = maxId+1
			fout.write("%d,%d,%d\n" % (i, maxId+1, self.sub[i+self.numTrain]))
			#if maxId+1 == self.sub[i+self.numSub]

	def dnn(self):
		lm = Sequential()
		lm.add(Dense(input_dim = self.dimWordVector, output_dim = self.n, W_regularizer = l2(self.r)))
		lm.add(Activation('sigmoid'))
		lm.add(Dropout(self.d))
		lm.add(Dense(output_dim = self.n, W_regularizer = l2(self.r)))
		lm.add(Activation('sigmoid'))
		lm.add(Dropout(self.d))
		lm.add(Dense(output_dim = self.numSub))
		lm.add(Activation('softmax'))
		lm.summary()

		lm.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		lm.fit(self.x, self.y, batch_size = self.batch, nb_epoch = self.epoch, shuffle = True)
		score, acc = lm.evaluate(self.x_test, self.y_test)
		#result = lm.predict(self.x_test)
		#self.output(result)

		print 'Test score:', score
		print 'Test accuracy:', acc

	def rnn(self):
		lm = Sequential()
		#lm.add(SimpleRNN(100, input_dim = self.dimWordVector, input_length = self.length, W_regularizer = l2(0.01), U_regularizer = l2(0.01), b_regularizer = l2(0.01)))
		lm.add(SimpleRNN(100, input_dim = self.dimWordVector, input_length = self.length, W_regularizer = l2(self.r), 
			U_regularizer = l2(self.r), b_regularizer = l2(self.r), return_sequences = True))
		lm.add(Dropout(0.25))
		lm.add(SimpleRNN(100, W_regularizer = l2(self.r), U_regularizer = l2(self.r), b_regularizer = l2(self.r)))
		lm.add(Dropout(0.25))
		lm.add(Dense(output_dim = self.numSub))
		lm.add(Activation('softmax'))
		lm.summary()

		lm.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		lm.fit(self.x, self.y, batch_size = self.batch, nb_epoch = self.epoch, shuffle = True)
		score, acc = lm.evaluate(self.x_test, self.y_test)
		#result = lm.predict(self.x_test)
		#self.output(result)

		print 'Test score:', score
		print 'Test accuracy:', acc

	def lstm(self):
		lm = Sequential()
		#lm.add(LSTM(100, input_dim = self.dimWordVector, input_length = self.length, W_regularizer = l2(0.01), U_regularizer = l2(0.01), b_regularizer = l2(0.01)))
		lm.add(LSTM(100, input_dim = self.dimWordVector, input_length = self.length, W_regularizer = l2(self.r), 
			U_regularizer = l2(self.r), b_regularizer = l2(self.r), return_sequences = True))
		lm.add(Dropout(0.25))
		lm.add(LSTM(100, W_regularizer = l2(self.r), U_regularizer = l2(self.r), b_regularizer = l2(self.r)))
		lm.add(Dropout(0.25))
		lm.add(Dense(output_dim = self.numSub))
		lm.add(Activation('softmax'))
		lm.summary()

		lm.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		lm.fit(self.x, self.y, batch_size = self.batch, nb_epoch = self.epoch, shuffle = True)
		score, acc = lm.evaluate(self.x_test, self.y_test)
		#result = lm.predict(self.x_test)
		#self.output(result)

		print 'Test score:', score
		print 'Test accuracy:', acc

	def gru(self):
		lm = Sequential()
		#lm.add(GRU(100, input_dim = self.dimWordVector, input_length = self.length, W_regularizer = l2(0.03), U_regularizer = 0.01, b_regularizer = 0.01))
		lm.add(GRU(100, input_dim = self.dimWordVector, input_length = self.length, W_regularizer = l2(self.r), 
			U_regularizer = l2(self.r), b_regularizer = l2(self.r), return_sequences = True))
		lm.add(Dropout(0.25))
		lm.add(GRU(100, W_regularizer = l2(self.r), U_regularizer = l2(self.r), b_regularizer = l2(self.r)))
		lm.add(Dropout(0.25))
		lm.add(Dense(output_dim = self.numSub))
		lm.add(Activation('softmax'))
		lm.summary()

		lm.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		lm.fit(self.x, self.y, batch_size = self.batch, nb_epoch = self.epoch, shuffle = True)
		score, acc = lm.evaluate(self.x_test, self.y_test)
		#result = lm.predict(self.x_test)
		#self.output(result)

		print 'Test score:', score
		print 'Test accuracy:', acc

	def test(self):
		for i in range(10):
			print self.sub[i], self.titleVector[i]

	def count(self):
		m = 0
		for i in range(len(self.doc)):
			if len(self.doc[i]) > m:
				m = len(self.doc[i])
		print "Longest = ", m


model = Model()
model.load()
model.wordVector(0)

if model.method == 0: # dnn
	model.aveEmbedding(1)
	model.setXY()
	model.n = 100
	model.r = 0.01
	model.d = 0
	model.dnn()

else:
	model.embedding()
	model.setXY()
	model.r = 0.01
	for i in range(3):
		model.method = i+1
		if model.method == 1: # rnn
			model.rnn()
		elif model.method == 2: # lstm
			model.lstm()
		elif model.method == 3: # gru
			model.gru()
	
	

