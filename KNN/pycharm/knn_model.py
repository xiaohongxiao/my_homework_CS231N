import numpy as np

class KNearestNeighbor:
    def __init__(self):
        pass
    def train(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k = 1, num_loops = 0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k = k)

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]  #test picture data
        num_train = self.X_train.shape[0] #train picture data
        between_distance = np.zeros((num_test, num_train))

        test_sum = np.sum(np.square(X), axis= 1)
        train_sum = np.sum(np.square(self.X_train),axis = 1)
        inner_product = np.dot(X, self.X_train.T)  #

        between_distance = np.sqrt(-2 * inner_product + test_sum.reshape(-1,1) + train_sum)
        return between_distance

    # 将距离排序，输出与测试集距离最小的前K个训练集图像的类别
    def predict_labels(self,distance, k = 1):
        num_test = distance.shape[0]
        y_predict = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            y_indicies = np.argsort(distance[i,:], axis = 0)
            print('length = ',len(y_indicies))
            print('y_indicies =' , y_indicies)
            closest_y = self.y_train[y_indicies[0:k]]  #选择k值很重要
            print('closest_y = ', closest_y)
            y_predict[i] = np.argmax(np.bincount(closest_y))
        return y_predict

