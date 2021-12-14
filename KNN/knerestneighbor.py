import numpy as numpy

class KNerarestNeighbor():
    def __init__(self,k):
       self.k=k

    def train(self,X,y):
        self.X_train=X
        self.y_train=y

    def predict(self,X_test,num_loops=2):
        if num_test==2:
            distances=self.compute_distance(X_test)
            return self.predict_labels(distances)
        if num_loops==1:
            distances=self.compute_distance_one_loop(X_test)
            return self.predict_labels(distances)
        else:
            distances=self.compute_distance_vectorized(X_test)



    def compute_distance_one_loop(self,X_test):
        num_test=X_test.shape[0]
        num_train=self.X_train.shape[0]
        distances=np.zeros((num_test,num_train))

        for i in range(num_test):
            distances[i,:]=np.sqrt(np.sum(((self.X_train-X_test[i,:])**2),axis=1))
            #按行相加，每一行是一个特征
            return distances
            


    def compute_distance_two_loop(self,X_test):
        #naive ways
        num_test=X_testrain.shape[0]
        num_train=self.X_train.shape[0]
        distances=np.zeros((num_test,num_train))

        for i in range (num_test):
            for j in range(num_test):
                distances[i,j]=np.sqrt(np.sum((X_test[i,:]-self.X_train[j,:])**2))
        return distances


    #train,test
    #(train-test)^2 =train^2-2train*test+test^2

    def predict_labels(self,distances):
        num_test=distances.shape[0]
        y_pred=np.zeros(num_test)
        
        for i in range(num_test):
            y_indices=np.argsort(distances[i,:])
            k_closest_classes=self.y_train[y_indices[:self.k]].astype(int)
            y_pred[i]=np.argmax(np.bincount(k_closest_classes))
            #bincount直接记录的是自然数的个数，从0开始

        return y_pred


    if__name__=='__main__':
        #X=np.loadtxt('')
        #y=np.loadtxt('')
        train=np.random.randn(1,4)
        test=np.random.randn(1,4)
        num_examples=train.shape[0]

        distance=np.sqrt(np.sum(test**2,axis=1,keepdims=True)+np.sum(train**2,keepdims=True,axis=1)
        -2*np.sum(test*train))
        #keepdims防止python改变格式，如将（1，1）变成（1，）
        KNN=KNerarestNeighbor(k=3)
        KNN.train(train,np.zeros((num_examples)))
        corr_distance=KNN>compute_distance_two_loop(test)
        
        print(f'Thedifference is :{np.sum(np.sum((corr_distance-distance)**2))}')
        #y_pred=KNN.predict_labels(X)

        #print(f'Accuracy:{sum(y_pred==y)/y.shape[0]}')

