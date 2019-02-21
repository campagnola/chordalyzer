import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


labels = np.load('labels.npy')
features = np.load('features.npy')[:, :300]
features /= features.max()
features -= features.mean()

n_examples, n_features = features.shape
n_classes = labels.shape[1]


def model():
	model = Sequential()
	model.add(Dense(100, input_dim=n_features, activation='relu'))
	# model.add(Dense(50, input_dim=n_features, activation='relu'))
	model.add(Dense(n_classes, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

classifier = KerasClassifier(build_fn=model, epochs=20, batch_size=10, verbose=1)
classifier.fit(features, labels)
# kfold = KFold(n_splits=3, shuffle=True, random_state=0)

# results = cross_val_score(classifier, features, labels, cv=kfold, verbose=1)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



