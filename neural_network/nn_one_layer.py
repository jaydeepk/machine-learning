
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


def get_data(one_hot_encoding=True):
    (train_images, train_labels), (test_images, test_labels) =  mnist.load_data()

    train_images = train_images.reshape((60000, 28*28))
    train_images = train_images.astype('float32')/255

    test_images = test_images.reshape((10000, 28*28))
    test_images = test_images.astype('float32')/255

    if one_hot_encoding is True:
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def build_neural_network(one_hot_encoding):
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    if one_hot_encoding is True:
        network.add(layers.Dense(10, activation='softmax'))
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        network.add(layers.Dense(1, activation='softmax'))
        network.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return network


def run(one_hot_encoding):
    train_images, train_labels, test_images, test_labels = get_data(one_hot_encoding)
    network = build_neural_network(one_hot_encoding)
    network.fit(train_images, train_labels, epochs=10, batch_size=128)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test accuracy:', test_acc)


if __name__ == "__main__":
    run(one_hot_encoding=True)
