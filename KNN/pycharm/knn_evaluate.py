import numpy as np
from knn_train import load_cifar10
import matplotlib.pyplot as plt
from knn_model import KNearestNeighbor
def main():
    x_train, y_train, x_test, y_test = load_cifar10('E:\cs231n-homework\KNN\cifar10batchespy')
    print('training data shape', x_train.shape)
    show(y_train,x_train)

    x_train, y_train, x_test, y_test = fast_speed_train(x_train, y_train, x_test, y_test)
    # x_train1, x_test1 = data_reshape(x_train, x_test)
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    test_set_predict(x_train, y_train, x_test, y_test)



def show(y_train,x_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls, in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        print("idxs = ", idxs)
        idxs = np.random.choice(idxs, samples_per_class, replace = False)
        print("idxs1 = ", idxs)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(x_train[idx].astype('uint8'))
            plt.axis('off')
            if i==0:
                plt.title(cls)
    plt.show()


def fast_speed_train(x_train, y_train, x_test,y_test):
    '''
    选取5000张训练集， 500张测试集，加快训练
    :return:
    '''
    number_training = 5000
    mask = range(number_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    num_test = 500
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]
    return x_train, y_train,x_test, y_test


def data_reshape(x_train, x_test):
    '''
    把单个数据拉成行向量，方便距离计算
    5000个数据，因此是5000* （32*32*3）
    :param x_train:
    :param x_test:
    :return:
    '''
    x_train = np.reshape(x_train,(x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    print(x_train.shape,'  ', x_test.shape)


def test_set_predict(x_train, y_train,x_test, y_test):
    classifier = KNearestNeighbor()
    classifier.train(x_train, y_train)
    distance = classifier.compute_distances_no_loops(x_test)
    print(distance)
    # 预测测试集类别
    y_test_predict = classifier.predict_labels(distance, k = 10)

    #使用准确率作为模型评价指标
    num_correct = np.sum(y_test_predict == y_test)
    accurancy = float(num_correct) / 500
    print('got %d / %d correct, accuracy: %.3f'%(num_correct, 500, accurancy))






if __name__ == '__main__':
    main()


