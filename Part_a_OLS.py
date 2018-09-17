
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import randrange, uniform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2)- 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

# Ordinary Least Square on the Franke function.
#def OLS():

#Defining training data

xtrain = np.random.permutation(np.arange(0, 1, 0.05)).reshape(-1, 1) #(20,1)
print(np.shape(xtrain))
ytrain = np.random.permutation(np.arange(0, 1, 0.05)).reshape(-1, 1) #(20,1)
#xtrain, ytrain = np.meshgrid(xtrain,ytrain)

#Making a model from Frankes function
z = FrankeFunction(xtrain, ytrain)
print("z")
print(np.shape(z)) #(20,20)

xyb = np.c_[np.ones((20)), xtrain, ytrain, xtrain*xtrain, ytrain*ytrain, xtrain*ytrain] #(20,6)
print("xyb")
print (np.shape(xyb)) # (20,6)
beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)
print("beta:")
print(np.shape(beta))   #(6,1)

fig = plt.figure()
ax = fig.gca(projection='3d')

#Chose test data.
xtest = np.random.permutation(np.arange(0, 1, 0.05)).reshape(-1, 1) #(20,1)
ytest = np.random.permutation(np.arange(0, 1, 0.05)).reshape(-1, 1) #(20,1)
#xtest, ytest = np.meshgrid(xtest,ytest)


xybnew = np.c_[np.ones(20), xtest, ytest, xtest*xtest, ytest*ytest, xtest*ytest]
print("xybnew")
print(np.shape(xybnew))
zpredict = xybnew.dot(beta) #(20, 101)

#Plot surface - predicted
xtest, ytest = np.meshgrid(xtest,ytest)
surfpredict = ax.plot_surface(xtest,ytest,zpredict,cmap=cm.coolwarm, linewidth=0, antialiased=False)

#Customise the z axis
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormaatter('%.02f'))


#Add a color bar which maps values to colors
fig.colorbar(surfpredict, shrink=0.5, aspect=5)
plt.show()
