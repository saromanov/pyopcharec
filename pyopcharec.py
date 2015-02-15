import numpy as np
from sklearn import metrics, svm, decomposition
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from skimage import io, filter
from skimage.transform import resize
from scipy import ndimage
from sklearn.cross_validation import train_test_split

def load_digits():
	return datasets.load_digits()

def load_mnist():
	return fetch_mldata('MNIST original', data_home='../datasets')

#http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
def prepare_characters(path):
	coll = io.ImageCollection('../English/Hnd/Img/Sample005/*.png')
	print(coll[10])

def SVM_train(data, target):
	train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.33, random_state=42)
	classifier = svm.SVC(gamma=0.001)
	classifier.fit(train_data, train_labels)
	predicted = classifier.predict(test_data)
	lens = len(predicted)
	cost = 0
	for i in range(lens):
		cost += (predicted[i] == test_labels[i])
	print(cost/len(test_labels), cost, len(test_labels))

def SVM_train_without_splitting(data, test_data, test_labels):
	"""
		data is digits from dataset
	"""
	classifier = svm.SVC(gamma=0.001)
	classifier.fit(data.data, data.target)
	predicted = classifier.predict(test_data)
	lens = len(predicted)
	cost = 0
	for i in range(lens):
		cost += (predicted[i] == test_labels[i])
	print(cost/len(test_labels), cost, len(test_labels))

def toy_example():
	data = load_digits()
	print("Clear data. Train with SVM")
	SVM_train(data.data, data.target)
	print("After blurred. Train with SVM")
	data_blurred, labels = noise_images(data, 1200)
	SVM_train(data_blurred, labels)
	test_blurred, lables_blurred = noise_images_as_test(data, 700)
	print("After blurred test data. Train with SVM")
	SVM_train_without_splitting(data, test_blurred, lables_blurred)

def split_data(data, pr=.85):
	"""
		p - percent of training data
		return: traindata, testdata
		where train([data], label)
	"""
	value = int(len(data.target) * pr)
	rang = range(len(data.target))
	trainidx = np.random.choice(rang, value, replace=False)
	testidx = [i for i in rang if i not in trainidx.tolist()]
	pre = [(img, target) for img, target in zip(data.data, data.target)]
	return np.take(pre, trainidx, axis=0).tolist(), np.take(pre, testidx, axis=0).tolist()


def noise_images(data, num_examples):
	"""
		Noise images and append to dataset
	"""
	if num_examples  == 0 or num_examples > len(data.data):
		return
	result = data.data
	labels = data.target
	for img in range(num_examples):
		gaussian = ndimage.gaussian_filter(data.data[img], sigma=3)
		result = np.vstack((result,gaussian))
		labels = np.hstack((labels, data.target[img]))
	return result, labels


def noise_images_as_test(data, num_examples):
	"""
		Blurred images without append to dataset
	"""
	if num_examples  == 0 or num_examples > len(data.data):
		return
	result = np.ndarray((data.data[0].shape))
	labels = np.ndarray([])
	for img in range(num_examples):
		gaussian = ndimage.gaussian_filter(data.data[img], sigma=3)
		result = np.vstack((result,gaussian))
		labels = np.hstack((labels, data.target[img]))
	return result, labels
toy_example()