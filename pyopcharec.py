import numpy as np
from sklearn import metrics, svm, decomposition
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from skimage import data, io, filter
from skimage.transform import resize

def load_digits():
	return datasets.load_digits()

def load_mnist():
	return fetch_mldata('MNIST original', data_home='../datasets')

#http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
def prepare_characters(path):
	coll = io.ImageCollection('../English/Hnd/Img/Sample005/*.png')
	print(coll[10])


def toy_example():
	data = load_digits()
	train, test = split_data(data)
	pca = decomposition.PCA(n_components=3)
	data_pca = pca.fit_transform(data.data)
	classifier = svm.SVC(gamma=0.001)
	classifier.fit(data_pca, data.target)
	cost = 0
	for i in range(len(test)):
		cost += classifier.predict(test[i][0])[0] == test[i][1]
	print(cost/len(test), cost, len(test))

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
