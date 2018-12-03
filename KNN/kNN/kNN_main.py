import numpy as np



class NearestNeighbor():
	def __init__(self):
		pass

	def train(self, x, y):
		"""
		x 是 N x D 中的每一行都是一个例子
		y 是 一维向量大小为N 
		"""
		# 最近邻分类器，只简单记住所有训练数据
		self.Xtr = x
		self.Ytr = y

	def predict(self,x,k = 1):
		"""
		x 是N x D 其中的每一行是一个图像向量，希望去预测标签
		"""

		num_test = x.shape[0]

		# 确保输入类型和输出类型相互一只
		Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)

		# 循环所有测试行（图片）
		for i in  range(num_test):
			# 寻找与第i张测试图片最近的训练图片
			# 使用L2距离（两幅像素差的平方和的开根号）
			# distances = np.sqrt(np.sum(np.square(self.Xtr - x[i,:]),axis = 1))

			# 使用L1距离(两个图像之间的距离加绝对值)
			distances = np.sum(np.abs(self.Xtr - x[i,:]),axis = 1)
			# print("distances = ", distances)
			# print("type = ", distances.shape)
			# 对最K小距离进行排序（返回从小到大的索引值）  [3, 1, 2]
			k_min_index = np.argsort(distances, axis = 0)
			
		
			# print('g_index = ', np.type(g_index))
			# print('Ytr = ', self.Ytr[:k])
			# 得到 最近图片的标签
			Ypred[i] = np.argmax(np.bincount(self.Ytr[:k]))

		return Ypred



# ################################################
# --*coding=utf-8 *--
import pickle
# import numpy as np
import os

def load_cifar_batch(filename):
    with open(filename,'rb') as f:
        datadict = pickle.load(f,encoding='bytes')
        x = datadict[b'data']
        y = datadict[b'labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        y = np.array(y)
        return x,y


def load_cifar10(root):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d'%b )
        x, y = load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
    Xtrain = np.concatenate(xs)
    Ytrain = np.concatenate(ys)
    del x, y  # delete 引用次数
    Xtest, Ytest = load_cifar_batch(os.path.join(root, 'test_batch'))
    return Xtrain, Ytrain, Xtest, Ytest

# ##########################################################################






def  main():
	# 数据预处理
	# 数据清洗，载入数据
	Xtr, Ytr, Xte, Y_te = load_cifar10('E://artifical_inteligence_data//data_set//cifar-10-python//cifar-10-batches-py')  # root_path：E:\artifical_inteligence_data\data_set\cifar-10-python\cifar-10-batches-py

	# 50000张训练集中选取训练集2000张 减少计算量（测试结果是电脑带不动）
	# 10000张测试集中选取测试集100张 减少计算量
	# print(Xtr)
	Xtr = Xtr[:2000,:,:,:]
	Ytr	= Ytr[:2000]
	Xte = Xte[:100,:,:,:] 
	Y_te = Y_te[:100]



	# 一张图像拉成一个列向量， 共有shape(0) == 5000
	# print(Xtr.shape[0])
	Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
	
	Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)


	# 使用交叉验证,验证集的作用主要是去找最佳超参数
	Xval_rows = Xtr_rows[:400,:]
	Yval_rows = Ytr[:400]
	Xtr_rows = Xtr_rows[400:,:]
	Ytr = Ytr[400:]



	# 寻找可以在验证集上最佳的超参数
	validation_accurancies = [] #初始化验证的准确率
	for k in [1, 3, 5, 10, 20, 50, 100]:

		# 创建一个最近邻分类器实例
		knn = NearestNeighbor()
		# 用训练集的图像和标签，训练分类器
		knn.train(Xtr_rows, Ytr)
		# 预测测试集的标签
		Yval_predict = knn.predict(Xval_rows, k = k)
		acc = np.mean(Yval_predict == Yval_rows)
		print('accurancy:%f' %(acc, ))
		validation_accurancies.append((k, acc))
	# 小结：
	# 使用L2距离，相较L1距离 低2个百分点

	# 取出最准确的超参数K
	k_most_cur = validation_accurancies[(np.argmax(validation_accurancies,axis = 0))[1]][0]
	knn.predict(Xte_rows,k = k_most_cur)
	acc = np.mean(Yval_predict == Yval_rows)
	print('most accurancy:%f' %(acc))
if __name__ == '__main__':
	main()