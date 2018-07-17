
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_data():
    df = pd.read_csv('data/train.csv')
    x = df.drop('label', axis=1)
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train.values, y_train.values, x_test.values, y_test.values


def initialize_weights_and_bias(dim, output_classes_count):
    w = np.random.rand(dim, output_classes_count) * 0.01
    b = np.zeros((1, output_classes_count))
    return w, b


def calculate_gradients(probs, x, y, n, w, lamda):
    probs[range(n), y] -= 1
    probs /= n
    dw = np.dot(x.T, probs)
    dw += lamda * w
    db = np.sum(probs, axis=0, keepdims=True)
    return dw, db


def calculate_accuracy(x, y, w, b):
    scores = np.dot(x, w) + b
    predicted_class = np.argmax(scores, axis=1)
    return np.mean(predicted_class == y) * 100


def predict(x, w, b):
    z = np.dot(x, w) + b
    return np.argmax(z, axis=1)


def forward_prop(x, w, b):
    return sigmoid(np.dot(x, w) + b)


def model(x, y, alpha=0.1, lamda=1e-3, iteration_count=2000):
    n = x.shape[0]
    w, b = initialize_weights_and_bias(x.shape[1], 10)

    for i in range(iteration_count):
        z = np.dot(x, w) + b
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        log_probs = -np.log(probs[range(n), y])

        loss = (1 / n) * np.sum(log_probs)
        regularization_loss = 0.5 * lamda * np.sum(w * w)
        loss += regularization_loss

        if i % 100 == 0:
             print("iteration %d: loss %f" % (i, loss))

        dw, db = calculate_gradients(probs, x, y, n, w, lamda)
        w += -alpha * dw
        b += -alpha * db

    return w, b


def predict_test_data(w, b):
    x = pd.read_csv('data/test.csv')
    n = x.values.shape[0]
    predictions = predict(x.values, w, b)
    indexes = list(range(1, n + 1))
    csv = np.column_stack((np.array(indexes), predictions))
    np.savetxt('prediction.csv', csv, fmt='%d', delimiter=',', header="ImageId,Label")


def run():
    x_train, y_train, x_val, y_val = get_data()
    x_train = x_train/255.0
    x_val = x_val/255.0

    w, b = model(x_train, y_train)

    accu_tr = calculate_accuracy(x_train, y_train, w, b)
    print('training  set accuracy: %.2f' % accu_tr)

    accu_val = calculate_accuracy(x_val, y_val, w, b)
    print('validation set accuracy: %.2f' % accu_val)

    predict_test_data(w, b)



if __name__ == "__main__":
    run()

