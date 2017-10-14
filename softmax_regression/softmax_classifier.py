import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_mnist_data():
    df = pd.read_csv('train.csv')
    X = df.drop('label', axis=1)
    Y = df['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train.values, Y_train.values, X_test.values, Y_test.values


def initialize_weights_and_bias(dim, output_classes_count):
    W = np.random.rand(dim, output_classes_count) * 0.01
    b = np.zeros((1, output_classes_count))
    return W,b


def calculate_gradients(probs, n, Y):
    probs[range(n), Y] -= 1
    probs /= n
    dW = np.dot(X_train.T, probs)
    db = np.sum(probs, axis=0, keepdims=True)
    return dW, db


def optimize(X_train, Y_train, W, b, alpha=0.01, iter_count=2000):
    for i in range(iter_count):
        Z = np.dot(X_train, W) + b
        exp_z = np.exp(Z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        log_probs = -np.log(probs[range(n), Y_train])
        loss = (1 / n) * np.sum(log_probs)

        if i % 100 == 0:
            print("iteration %d: loss %f" % (i, loss))

        dW, db = calculate_gradients(probs, n, Y_train)
        W += -alpha * dW
        b += -alpha * db


def calculate_accuracy(X_test, Y_test, W, b):
    scores = np.dot(X_test, W) + b
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == Y_test) * 100))


def predict(X, W, b, n):
    Z = np.dot(X, W) + b
    predictions = np.argmax(Z, axis=1)
    indexes = list(range(1, n+1))
    csv = np.column_stack((np.array(indexes), predictions))
    np.savetxt('pred.csv', csv, fmt='%d', delimiter=',', header="ImageId,Label")


X_train, Y_train, X_test, Y_test = get_mnist_data()
X_train = X_train/255
X_test = X_test/255
W, b = initialize_weights_and_bias(X_train.shape[1], 10)
alpha = 0.05
n = X_train.shape[0]

optimize(X_train, Y_train, W, b, alpha)
calculate_accuracy(X_test, Y_test, W, b)

X = pd.read_csv('test.csv')
predict(X.values, W, b, X.values.shape[0])






