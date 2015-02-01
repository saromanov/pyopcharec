import numpy as np
from sklearn import metrics, svm, decomposition
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

def load_digits():
	return datasets.load_digits()

def split_data(data, pr=.75):
	"""
		p - percent of training data
	"""
	value = int(len(data.target) * pr)
	rang = range(len(data.target))
	trainidx = np.random.choice(rang, value, replace=False)
	testidx = [i for i in rang if i not in trainidx.tolist()]
	pre = [(img, target) for img, target in zip(data.data, data.target)]
	return np.take(pre, trainidx, axis=0).tolist(), np.take(pre, testidx, axis=0).tolist()

data = load_digits()
train, test = split_data(data)
pca = decomposition.PCA()
classifier = svm.SVC(gamma=0.001)
classifier.fit(list(map(lambda x: x[0], train)),  list(map(lambda x: x[1], train)))
cost = 0
for i in range(len(test)):
	cost += classifier.predict(test[i][0])[0] == test[i][1]
print(cost/len(test), cost, len(test))