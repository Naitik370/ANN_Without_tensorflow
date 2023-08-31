import numpy as np
import matplotlib.pyplot as plt
class ANN:
    def __init__(self,X,Y,n_h,activation_fn):
        self.X = X
        self.Y = Y
        self.n_x = X.shape[0]
        self.n_h = n_h
        self.n_y = Y.shape[0]
        self.activation_fn = activation_fn
    def activ_fn(self,fn,z,d=False):
        if(fn=="sig"):
            if d:
                return z*(1-z)
            else:
                return 1/(1+np.exp(-z))
        if(fn=="tan"):
            if d:
                return 1-z**2
            else:
                return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        if(fn=="relu"):
            if d:
                return np.int64(z>0)
            else:
                return np.maximum(0,z)
        if(fn=="lrelu"):
            if d:
                return np.where(z>0, 1, 0.01)
            else:
                return np.maximum(0.01,z)
    # def softmax(z,d = False):
    #     e = np.exp(z)
    #     e = e/np.sum()
    #     return s
    def initz(self):
        n_h = self.n_h
        np.random.seed(2)
        parameters = {}
        for i in range(len(n_h)):
            if(i==0):
                parameters['w1'] = np.random.randn(n_h[i],self.n_x)*0.01
                parameters['b1'] = np.zeros((n_h[i],1))
                continue
            parameters[f'w{i+1}']= np.random.randn(n_h[i],n_h[i-1])*0.01
            parameters[f'b{i+1}']= np.random.randn(n_h[i],1)*0.01
        print(parameters)
        return parameters
    def fwd(self,parameters):
        cache = {}
        for i in range(len(self.n_h)):
            if(i == 0):
                cache[f"Z{i+1}"] = np.dot(parameters[f"w{i+1}"],self.X)+parameters[f"b{i+1}"]
                cache[f"A{i+1}"] = self.activ_fn(self.activation_fn[i],cache[f"Z{i+1}"])
                continue
            cache[f"Z{i+1}"] = np.dot(parameters[f"w{i+1}"],cache[f"A{i}"])+parameters[f"b{i+1}"]
            cache[f"A{i+1}"] = self.activ_fn(self.activation_fn[i],cache[f"Z{i+1}"])
        print(cache)
        return cache
    def compute_cost(self,cache):
        m = self.Y.shape[1]
        A = cache[f"A{len(self.n_h)}"]
        # logp = np.multiply(np.log(A),self.Y)+np.multiply(np.log(1-A),(1-self.Y))
        logp = (np.log(A)*self.Y)+(np.log(1-A)*(1-self.Y))
        cost = -np.sum(logp)/m
        cost = float(np.squeeze(cost))
        return cost
    def bwp(self,parameters,cache):
        gradients = {}
        m = self.Y.shape[1]
        i = len(self.n_h)-1
        while(i>=0):
            if(i==(len(self.n_h)-1)):
                gradients[f"dZ{i+1}"] = cache[f"A{i+1}"]-self.Y
                gradients[f"dw{i+1}"] = np.dot(gradients[f"dZ{i+1}"],cache[f"A{i}"].T)/m
                gradients[f"db{i+1}"] = np.sum(gradients[f"dZ{i+1}"],axis=1,keepdims=True)/m
                i=i-1
                continue
            elif(i==0):
                gradients[f"dA{i+1}"] = np.dot(parameters[f"w{i+2}"].T,gradients[f"dZ{i+2}"])
                gradients[f"dZ{i+1}"] = gradients[f"dA{i+1}"]*self.activ_fn(self.activation_fn[i],cache[f"Z{i+1}"],d=True)
                gradients[f"dw{i+1}"] = np.dot(gradients[f"dZ{i+1}"],self.X.T)/m
                gradients[f"db{i+1}"] = np.sum(gradients[f"dZ{i+1}"],axis=1,keepdims=True)/m
                i=i-1
                continue
            gradients[f"dA{i+1}"] = np.dot(parameters[f"w{i+2}"].T,gradients[f"dZ{i+2}"])
            gradients[f"dZ{i+1}"] = gradients[f"dA{i+1}"]*self.activ_fn(self.activation_fn[i],cache[f"Z{i+1}"],d=True)
            gradients[f"dw{i+1}"] = np.dot(gradients[f"dZ{i+1}"],cache[f"A{i}"].T)/m
            gradients[f"db{i+1}"] = np.sum(gradients[f"dZ{i+1}"],axis=1,keepdims=True)/m
            i=i-1
        print(gradients)
        return gradients
    def update(self,gradients,parameters,lr=0.01):
        for i in range(len(self.n_h)):
            parameters[f"w{i+1}"] = parameters[f"w{i+1}"] - lr*gradients[f"dw{i+1}"]
            parameters[f"b{i+1}"] = parameters[f"b{i+1}"] - lr*gradients[f"db{i+1}"]
        print(parameters)
        
        return parameters

def main():
    np.random.seed(1)
    X=np.random.randn(2,3)
    Y=np.int64(np.random.randn(1,3)>0)
    ob = ANN(X,Y,[4,2,1],['relu','lrelu','sig'])
    param = ob.initz()
    loss = []
    for i in range(2):
        cache = ob.fwd(param)
        cost = ob.compute_cost(cache)
        gradients = ob.bwp(param,cache)
        param = ob.update(gradients,param,0.01)
        loss.append(cost)
        if i%1==0:
            print("cost % i:%f" %(i,cost))
    plt.plot(loss)    
    
main()
