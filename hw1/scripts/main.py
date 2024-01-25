# %%
import sys

print(sys.version)
# %%
# This file is in scripts/load.py
import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
from tqdm import tqdm
if __name__ == "__main__":
    for data_name in ["mnist", "spam", "toy"]:
        data = np.load(f"../data/{data_name}-data.npz")
        print("\nloaded %s data!" % data_name)
        fields = "test_data", "training_data", "training_labels"
        for field in fields:
            print(field, data[field].shape)

np.random.seed(10072001)

# %%
def shuffle_data(data, labels):
    np.random.seed(10072001)
    p = np.random.permutation(len(data))
    sdata, slabels = data[p], labels[p]
    
    return sdata, slabels

# %%
def partition_data(data, labels, size):
    sdata, slabels = shuffle_data(data, labels)
    
    tdata, tlabels = sdata[:size-1], slabels[:size-1]
    tedata, telabels = sdata[size:], slabels[size:]
    
    return tdata, tlabels, tedata, telabels

# %%
def mnist_partition():
    # uses set seed above to determine random shuffling
    data = np.load(f"../data/mnist-data.npz")
    mnist_data = np.copy(data["training_data"])
    mnist_labels = np.copy(data["training_labels"])
    
    mnist_data = mnist_data.reshape(len(mnist_data), -1)

    val_data, val_labels, train_data, train_labels = partition_data(mnist_data, mnist_labels, 10000)
    
    return train_data, train_labels, val_data, val_labels

# %%
def spam_partition():
    # repeat for spam dataset with 20%
    data = np.load(f"../data/spam-data.npz")
    spam_data = np.copy(data["training_data"])
    spam_labels = np.copy(data["training_labels"])
    
    spam_data.reshape(len(spam_data), -1)

    perc = round(len(spam_data)/5)
    
    val_data, val_labels, train_data, train_labels = partition_data(spam_data, spam_labels, perc)
    
    return train_data, train_labels, val_data, val_labels

# %%
def accuracy(y, y_hat):
    assert len(y) == len(y_hat)
    n = len(y)
    accuracy = 0
    for i in np.arange(0, n):
        if y[i] == y_hat[i]:
            accuracy += 1
    return round(accuracy / n, 4)

# %%
def train_svm_mnist():
    data, data_labels, val, val_labels = mnist_partition()
    sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    train_accs = []
    val_accs = []
    
    for size in tqdm(sizes):
        tdata, tlabels, tedata, telabels = partition_data(data, data_labels, size)
    
        clf = svm.SVC(kernel="linear")
        clf.fit(tdata, tlabels)
        pred_labels = clf.predict(tdata)
        pred_val_labels = clf.predict(val)
    
        train_acc = accuracy(tlabels, pred_labels)
        val_acc = accuracy(val_labels, pred_val_labels)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    
    plt.plot(sizes, train_accs, label="Training Accuracy")
    plt.plot(sizes, val_accs, label="Validation Accuracy")
    plt.xlabel("Training Sizes")
    plt.ylabel("Accuracy")
    plt.title("MNIST Training Accuracies")
    plt.legend()
    plt.plot()

# %%
train_svm_mnist()

# %%
def train_svm_spam():
    data, data_labels, val, val_labels = spam_partition()
    sizes = [100, 200, 500, 1000, 2000, len(data)]
    train_accs = []
    val_accs = []
    
    for size in tqdm(sizes):
        tdata, tlabels, tedata, telabels = partition_data(data, data_labels, size)

        clf = svm.SVC(kernel="linear")
        clf.fit(tdata, tlabels)
        pred_labels = clf.predict(tdata)
        pred_val_labels = clf.predict(val)
    
        train_acc = accuracy(tlabels, pred_labels)
        val_acc = accuracy(val_labels, pred_val_labels)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    
    plt.plot(sizes, train_accs, label="Training Accuracy")
    plt.plot(sizes, val_accs, label="Validation Accuracy")
    plt.xlabel("Training Sizes")
    plt.ylabel("Accuracy")
    plt.title("Spam Training Accuracies")
    plt.legend()
    plt.show()

# %%
    
train_svm_spam()

# %%
def mnist_c_manipulation():
    d = np.load(f"../data/mnist-data.npz")
    data = d["training_data"]
    labels = d["training_labels"]
    data = data.reshape(len(data), -1)
    size = 10000
    results = []
    sees = [.001, .0005, .0001, .00005, .00001, .000005, .000001, .0000001]
    # sees = [1, 2, 3, 4, 5, 6]

    tdata, tlabels, val, val_labels = mnist_partition()
    tdata, tlabels, tedata, telabels = partition_data(tdata, tlabels, size)
    
    for c in tqdm(sees):
        # fit on full data set including validation data
        clf = svm.SVC(C=c, kernel="linear")
        clf.fit(tdata, tlabels)
        
        pred_val_labels = clf.predict(val)

        out = [c, accuracy(val_labels, pred_val_labels)]
        results.append(out)

    return results

# BEST RESULT C = 1e-6 and size = 10,000, ACC = .932
# BEST RESULT C = 1e-6 and size = 20,000, ACC = .9417

# %%
#mnist_c_manipulation()

# %%
def spam_c_manipulation():
    data = np.load(f"../data/spam-data.npz")
    spam_data = np.copy(data["training_data"])
    spam_labels = np.copy(data["training_labels"])

    spam_data.reshape(len(spam_data), -1)

    #sees = [1, .5, .1, .05, .01, .005, .001, .0001]
    sees = [1, 2, 5, 10, .05, .01, .005, .0001]
    # sees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # sees = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
    results = []

    sd, sl = shuffle_data(spam_data, spam_labels)

    chunk = round(len(sd)/5)

    for c in tqdm(sees):
        accs = []
        for i in np.arange(5):
            # sd[i] becomes val data, rest becomes training
            val = sd[chunk*i:chunk*(i+1)]
            val_labels = sl[chunk*i:chunk*(i+1)]

            train = np.concatenate([sd[:chunk*i], sd[chunk*(i+1):]], axis=0)
            train_labels = np.concatenate([sl[:chunk*i], sl[chunk*(i+1):]], axis=0)

            clf = svm.SVC(C=c, kernel="linear")
            clf.fit(train, train_labels)
            pred_val_labels = clf.predict(val)
            acc = accuracy(val_labels, pred_val_labels)

            accs.append(acc)

        avg_acc = round(np.mean(accs), 4)
        results.append([c, avg_acc])

    return results

## BEST RESULT IS C = 10 with acc .8036
## BEST RESULT IS C = 7 with acc .8211
## best result with rbf is c = 9000 with acc .8494

# %%
#spam_c_manipulation()


# %%
from save_csv import results_to_csv

d = np.load(f"../data/mnist-data.npz")
data = d["training_data"]
labels = d["training_labels"]
test = d["test_data"]

data = data.reshape(len(data), -1)
test = test.reshape(len(test), -1)

clf = svm.SVC(C=4, kernel="rbf")
clf.fit(data, labels)

results_to_csv(clf.predict(test))

# %%
from save_csv import results_to_csv

d = np.load(f"../data/spam-data.npz")
data = d["training_data"]
labels = d["training_labels"]
test = d["test_data"]

data = data.reshape(len(data), -1)
test = test.reshape(len(test), -1)

clf = svm.SVC(C=9000, kernel="rbf")
clf.fit(data, labels)

results_to_csv(clf.predict(test))