import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


labels = np.load('labels.npy')
features = np.load('features.npy')

n_examples, n_features = features.shape
n_classes = labels.shape[1]


def model():
	model = Sequential()
	model.add(Dense(1000, input_dim=n_features, activation='relu'))
	model.add(Dense(n_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

classifier = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

results = cross_val_score(classifier, features, labels, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

