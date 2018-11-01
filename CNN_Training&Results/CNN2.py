import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras.models as models
from keras.models import *
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
import matplotlib.pyplot as plt
import pickle
import keras
import sys
sys.path.append('../confusion')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score
import h5py

name = 'CNN2'

Xd = pickle.load(open("RML2016.10a_dict.dat", 'rb'), encoding='Latin-1')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
	for snr in snrs:
		X.append(Xd[(mod, snr)])
		for i in range(Xd[(mod, snr)].shape[0]):
			lbl.append((mod, snr))

X = np.vstack(X)
np.random.seed(36)  
n_examples = X.shape[0]
n_train = n_examples * 0.7
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx)) 
X_train = X[train_idx]
X_test = X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train = to_onehot(trainy)
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods
dr = 0.5 
model = models.Sequential() 
model.add(Reshape(([1] + in_shp), input_shape=in_shp))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(256, (1, 3), padding='valid', activation="relu", name="conv1", init='glorot_uniform',data_format="channels_first"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(256, (1, 3), padding="valid", activation="relu", name="conv2", init='glorot_uniform',data_format="channels_first"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", init='glorot_uniform',data_format="channels_first"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", init='glorot_uniform',data_format="channels_first"))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), init='he_normal', name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 2000
batch_size = 1024  
filepath = "conv_%s.h5" % (name) 

history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_test, Y_test))

plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.savefig('%s Training performance' %(name))

model.save(filepath)
model.save_weights('CNNmodel_weights.h5')
# we re-load the best weights once training is finished
mod = load_model(filepath)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print('evaluate_score:', score)

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
	if normalize:
		cm = ((cm.astype('float'))/(cm.sum(axis=1)[:, np.newaxis]))
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

test_Y_hat = model.predict(X_test, batch_size=batch_size)

pre_labels = []
for x in test_Y_hat:
    tmp = np.argmax(x, 0)
    pre_labels.append(tmp)
true_labels = []
for x in Y_test:
    tmp = np.argmax(x, 0)
    true_labels.append(tmp)

kappa = cohen_kappa_score(pre_labels, true_labels)
oa = accuracy_score(true_labels, pre_labels)
kappa_oa = {}
print('oa_all:', oa)
print('kappa_all:', kappa)
kappa_oa['oa_all'] = oa
kappa_oa['kappa_all'] = kappa
fd = open('results_all_%s_d0.5.dat' % (name), 'wb')
pickle.dump(("%s" % (name), 0.5, kappa_oa), fd)
fd.close()
cnf_matrix = confusion_matrix(true_labels, pre_labels)
plot_confusion_matrix(cnf_matrix, classes=classes,normalize=False,title='%s Confusion matrix, without normalization' % (name))
plt.savefig('%s Confusion matrix without normalization' % (name))

plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,title='%s Normalized confusion matrix' % (name))
plt.savefig('%s Normalized confusion matrix' % (name))

conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] += 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
plot_confusion_matrix(confnorm, classes=classes, title='%s Confusion matrix' % (name))

acc = {}
kappa_dict = {}
oa_dict = {}
for snr in snrs:
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i_hat = model.predict(test_X_i)
    pre_labels_i = []
    for x in test_Y_i_hat:
        tmp = np.argmax(x, 0)
        pre_labels_i.append(tmp)
    true_labels_i = []
    for x in test_Y_i:
        tmp = np.argmax(x, 0)
        true_labels_i.append(tmp)
    kappa = cohen_kappa_score(pre_labels_i, true_labels_i)
    oa = accuracy_score(true_labels_i, pre_labels_i)
    oa_dict[snr] = oa
    kappa_dict[snr] = kappa
    cnf_matrix = confusion_matrix(true_labels_i, pre_labels_i)
    plot_confusion_matrix(cnf_matrix, classes=classes,normalize=False,title='%s Confusion matrix, without normalization (SNR=%d)' % (name, snr))
    plt.savefig('%s Confusion matrix, without normalization (SNR=%d)' % (name, snr))
    plot_confusion_matrix(cnf_matrix, classes=classes,normalize=True,title='%s Normalized confusion matrix (SNR=%d)' % (name, snr))
    plt.savefig('%s Normalized confusion matrix (SNR=%d)' % (name, snr))
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] += 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plot_confusion_matrix(confnorm, classes=classes, title="%s Confusion Matrix (SNR=%d)" % (name, snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)

print('acc:', acc)
fd = open('results_%s_d0.5.dat' % (name), 'wb')
pickle.dump(("%s" % (name), 0.5, acc), fd)
fd.close()
print('oa:', oa_dict)
fd = open('results_oa_%s_d0.5.dat' % (name), 'wb')
pickle.dump(("%s" % (name), 0.5, oa_dict), fd)
fd.close()
print('kappa:', kappa_dict)
fd = open('results_kappa_%s_d0.5.dat' % (name), 'wb')
pickle.dump(("%s" % (name), 0.5, kappa_dict), fd)
fd.close()

plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("%s Classification Accuracy on RadioML 2016.10 Alpha" % (name))
plt.savefig("%s Classification Accuracy" % (name))
