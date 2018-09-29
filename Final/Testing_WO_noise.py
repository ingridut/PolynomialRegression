import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import randrange, uniform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


#FrankeFunction - for simulation of data
def FrankeFunction(x,y, noise):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2)- 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4 + noise*np.random.randn(x.shape[0], x.shape[1])
    
#Ordinary Least Squared function
def ols(x, y, z, degree = 5):
    xyb_ = np.c_[x, y]
    poly = PolynomialFeatures(degree)
    xyb = poly.fit_transform(xyb_)
    beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)
    return beta

#Mean Squared Error
def mse(zReal, zPredicted):
    mse = np.mean((zReal-zPredicted)**2)
    return mse           

def bias(zReal, zPredicted):
    bias = np.mean( (zReal- np.mean(zPredicted))**2 )
    return bias
    
def var(zReal, zPredicted):
    var = np.mean( (zPredicted - np.mean(zPredicted))**2 )
    return var

#Mean value of the function
def Mean(z):
    meanValue = np.mean(z)
    return meanValue

#R2 score function
def R2(zReal, zPredicted):
    meanValue = Mean(zReal)
    numerator = np.sum((zReal - zPredicted)**2)
    denominator = np.sum((zReal - meanValue)**2)
    result = 1 - (numerator/denominator)
    return result
###############################################################################
    
#For plotting
fig = plt.figure(figsize = (10,10))
ax = fig.gca(projection='3d')

#Uploading training data
xx = np.load("data_for_part_1.npy")
X = xx[:,0].reshape(1000,1)
Y = xx[:,1].reshape(1000,1)

#Franke function with different noise levels
z = FrankeFunction(X, Y, 0) # with zero noise
z001 = FrankeFunction(X, Y, 0.001) #with 0.001 noise
z01 = FrankeFunction(X, Y, 0.01) # with 0.01 noise
z1 = FrankeFunction(X, Y, 0.1) #with 0.1 noise
z2 = FrankeFunction(X, Y, 0.2) # with 0.2 noise
z3 = FrankeFunction(X, Y, 0.3) #with 0.3 noise
z4 = FrankeFunction(X, Y, 0.4) #with 0.4 noise

#Getting beta from ols-function - NOISE: 0 for degree 2,3,4,5
beta2 = ols(X, Y, z, 2)
beta3 = ols(X, Y, z, 3)
beta4 = ols(X, Y, z, 4)
beta5 = ols(X, Y, z, 5)

#Getting beta from ols-function - NOISE: 0.001 for degree 2,3,4,5
beta2001 = ols(X, Y, z001, 2)
beta3001  = ols(X, Y, z001, 3)
beta4001  = ols(X, Y, z001, 4)
beta5001  = ols(X, Y, z001, 5)

#Getting beta from ols-function - NOISE: 0.01 for degree 2,3,4,5
beta201 = ols(X, Y, z01, 2)
beta301 = ols(X, Y, z01, 3)
beta401 = ols(X, Y, z01, 4)
beta501 = ols(X, Y, z01, 5)

#Getting beta from ols-function - NOISE: 0.1 for degree 2,3,4,5
beta21 = ols(X, Y, z1, 2)
beta31 = ols(X, Y, z1, 3)
beta41 = ols(X, Y, z1, 4)
beta51 = ols(X, Y, z1, 5)

#Getting beta from ols-function - NOISE: 0.02 for degree 2,3,4,5
beta22 = ols(X, Y, z2, 2)
beta32 = ols(X, Y, z2, 3)
beta42 = ols(X, Y, z2, 4)
beta52 = ols(X, Y, z2, 5)

#Getting beta from ols-function - NOISE: 0.03 for degree 2,3,4,5
beta23 = ols(X, Y, z3, 2)
beta33 = ols(X, Y, z3, 3)
beta43 = ols(X, Y, z3, 4)
beta53 = ols(X, Y, z3, 5)

#Getting beta from ols-function - NOISE: 0.4 for degree 2,3,4,5
beta24 = ols(X, Y, z4, 2)
beta34 = ols(X, Y, z4, 3)
beta44 = ols(X, Y, z4, 4)
beta54 = ols(X, Y, z4, 5)


#Geretaing 1000 point from 0 to 1 for plotting
X_test = np.linspace(0, 1, 1000)
Y_test = np.linspace(0, 1, 1000)
X_test, Y_test = np.meshgrid(X_test,Y_test)
X_flat_t = X_test.reshape(-1,1)
Y_flat_t = Y_test.reshape(-1,1)

#Chose degrre 2, 3, 4, 5
xyb_ = np.c_[X_flat_t, Y_flat_t]
poly2 = PolynomialFeatures(2)
poly3 = PolynomialFeatures(3)
poly4 = PolynomialFeatures(4)
poly5 = PolynomialFeatures(5)

xyb2 = poly2.fit_transform(xyb_)
xyb3 = poly3.fit_transform(xyb_)
xyb4 = poly4.fit_transform(xyb_)
xyb5 = poly5.fit_transform(xyb_)

#Predicting Frankes function with zero noise and degree 2,3,4,5
zpredict2 = xyb2.dot(beta2).reshape(1000, 1000)
zpredict3 = xyb3.dot(beta3).reshape(1000, 1000)
zpredict4 = xyb4.dot(beta4).reshape(1000, 1000)
zpredict5 = xyb5.dot(beta5).reshape(1000, 1000)

#Predicting Frankes function with 0.001 noise and degree 2,3,4,5
zpredict2001 = xyb2.dot(beta2001).reshape(1000, 1000)
zpredict3001 = xyb3.dot(beta3001).reshape(1000, 1000)
zpredict4001 = xyb4.dot(beta4001).reshape(1000, 1000)
zpredict5001 = xyb5.dot(beta5001).reshape(1000, 1000)

#Predicting Frankes function with 0.01 noise and degree 2,3,4,5
zpredict201 = xyb2.dot(beta201).reshape(1000, 1000)
zpredict301 = xyb3.dot(beta301).reshape(1000, 1000)
zpredict401 = xyb4.dot(beta401).reshape(1000, 1000)
zpredict501 = xyb5.dot(beta501).reshape(1000, 1000)

#Predicting Frankes function with 0.1 noise and degree 2,3,4,5
zpredict21 = xyb2.dot(beta21).reshape(1000, 1000)
zpredict31 = xyb3.dot(beta31).reshape(1000, 1000)
zpredict41 = xyb4.dot(beta41).reshape(1000, 1000)
zpredict51 = xyb5.dot(beta51).reshape(1000, 1000)

#Predicting Frankes function with 0.2 noise and degree 2,3,4,5
zpredict22 = xyb2.dot(beta22).reshape(1000, 1000)
zpredict32 = xyb3.dot(beta32).reshape(1000, 1000)
zpredict42 = xyb4.dot(beta42).reshape(1000, 1000)
zpredict52 = xyb5.dot(beta52).reshape(1000, 1000)

#Predicting Frankes function with 0.3 noise and degree 2,3,4,5
zpredict23 = xyb2.dot(beta23).reshape(1000, 1000)
zpredict33 = xyb3.dot(beta33).reshape(1000, 1000)
zpredict43 = xyb4.dot(beta43).reshape(1000, 1000)
zpredict53 = xyb5.dot(beta53).reshape(1000, 1000)

#Predicting Frankes function with 0.4 noise and degree 2,3,4,5
zpredict24 = xyb2.dot(beta24).reshape(1000, 1000)
zpredict34 = xyb3.dot(beta34).reshape(1000, 1000)
zpredict44 = xyb4.dot(beta44).reshape(1000, 1000)
zpredict54 = xyb5.dot(beta54).reshape(1000, 1000)

#Z real with different noise levels
Z = FrankeFunction(X_flat_t, Y_flat_t, 0) #zero noise
Z001 = FrankeFunction(X_flat_t, Y_flat_t, 0.001) # 0.001 noise
Z01 = FrankeFunction(X_flat_t, Y_flat_t, 0.01) # 0.01 noise
Z1 = FrankeFunction(X_flat_t, Y_flat_t, 0.1) # 0.1 noise
Z2 = FrankeFunction(X_flat_t, Y_flat_t, 0.2) # 0.2 noise
Z3 = FrankeFunction(X_flat_t, Y_flat_t, 0.3) # 0.3 noise
Z4 = FrankeFunction(X_flat_t, Y_flat_t, 0.4) # 0.4 noise

#Z-REAL with different noise levels - RESHAPED
zreal= Z.reshape(1000, 1000)  #zero noise
zreal001 = Z001.reshape(1000, 1000) # 0.001 noise
zreal01 = Z01.reshape(1000, 1000) # 0.01 noise
zreal1 = Z1.reshape(1000, 1000)
zreal2 = Z2.reshape(1000, 1000)
zreal3 = Z3.reshape(1000, 1000)
zreal4 = Z4.reshape(1000, 1000)

#Plot surface - predicted
surfpredict2 = ax.plot_surface(X_test, Y_test,zpredict2,cmap=cm.coolwarm, linewidth=0, antialiased=False)
#surfpredict3 = ax.plot_surface(X_test, Y_test,zpredict3,cmap=cm.coolwarm, linewidth=0, antialiased=False)

#Plot Franke function REAL
#surfreal = ax.plot_surface(X_test, Y_test,zreal, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#points = ax.scatter( X_test, Y_test, ZZ) #Plot of the real values
    
#Customise the z axis
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
ax.set_title('Model of Frankes function');
ax.set_zlim(-0.10, 1.20)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#Add a color bar which maps values to colors
fig.colorbar(surfpredict2, shrink=0.5, aspect=5)
plt.show()

###############################################################################
print("MSE and R2 for noise = 0 and degree = 2, 3, 4, 5")
MSE = []
R2score = []
degree = [2, 3, 4, 5]
print ("Mean squared error (noise= 0, degree = 2): %.4f" % mse(zreal, zpredict2))
print("R2 score (noise= 0, degree = 2): %.4f" % R2(zreal, zpredict2))

MSE.append(mse(zreal, zpredict2))
R2score.append(R2(zreal, zpredict2))

print ("Mean squared error (noise= 0, degree = 3): %.4f" % mse(zreal, zpredict3))
print("R2 score (noise= 0, degree = 3): %.4f" % R2(zreal, zpredict3))
MSE.append(mse(zreal, zpredict3))
R2score.append(R2(zreal, zpredict3))

print ("Mean squared error (noise= 0, degree = 4): %.4f" % mse(zreal, zpredict4))
print("R2 score (noise= 0, degree = 4): %.4f" % R2(zreal, zpredict4))
MSE.append(mse(zreal, zpredict4))
R2score.append(R2(zreal, zpredict4))

print ("Mean squared error (noise= 0, degree = 5): %.4f" % mse(zreal, zpredict5))
print("R2 score (noise= 0, degree = 5): %.4f" % R2(zreal, zpredict5))
MSE.append(mse(zreal, zpredict5))
R2score.append(R2(zreal, zpredict5))

plt.plot(degree, MSE, label="MSE")
plt.plot(degree, R2score, label="R2 score")
plt.title('MSE and R2 score for model with noise = 0')
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
###############################################################################

###############################################################################
print("MSE and R2 for 0.001 noise and degree = 2, 3, 4, 5")
MSE001 = []
R2score001 = []
degree = [2, 3, 4, 5]

print ("Mean squared error (noise= 0.001, degree = 2): %.4f" % mse(zreal001, zpredict2001))
print("R2 score (noise= 0.001, degree = 2): %.4f" % R2(zreal001, zpredict2001))

MSE001.append(mse(zreal001, zpredict2001))
R2score001.append(R2(zreal001, zpredict2001))

print ("Mean squared error (noise= 0.001, degree = 3): %.4f" % mse(zreal001, zpredict3001))
print("R2 score (noise= 0.001, degree = 3): %.4f" % R2(zreal001, zpredict3001))

MSE001.append(mse(zreal001, zpredict3001))
R2score001.append(R2(zreal001, zpredict3001))

print ("Mean squared error (noise= 0.001, degree = 4): %.4f" % mse(zreal001, zpredict4001))
print("R2 score (noise= 0.001, degree = 4): %.2f" % R2(zreal001, zpredict4001))

MSE001.append(mse(zreal001, zpredict4001))
R2score001.append(R2(zreal001, zpredict4001))

print ("Mean squared error (noise= 0.001, degree = 5): %.4f" % mse(zreal001, zpredict5001))
print("R2 score (noise= 0.001, degree = 5): %.4f" % R2(zreal001, zpredict5001))

MSE001.append(mse(zreal001, zpredict5001))
R2score001.append(R2(zreal001, zpredict5001))


plt.plot(degree, MSE001, label="MSE")
plt.plot(degree, R2score001, label="R2 score")
plt.title('MSE and R2 score for model with noise = 0.001')
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
###############################################################################

###############################################################################
print("MSE and R2 for 0.01 noise and degree = 2, 3, 4, 5")
MSE01 = []
R2score01 = []
degree = [2, 3, 4, 5]

print ("Mean squared error (noise= 0.01, degree = 2): %.4f" % mse(zreal01, zpredict201))
print("R2 score (noise= 0.01, degree = 2): %.4f" % R2(zreal01, zpredict201))

MSE01.append(mse(zreal01, zpredict201))
R2score01.append(R2(zreal01, zpredict201))

print ("Mean squared error (noise= 0.01, degree = 3): %.4f" % mse(zreal01, zpredict301))
print("R2 score (noise= 0.01, degree = 3): %.4f" % R2(zreal01, zpredict301))

MSE01.append(mse(zreal01, zpredict301))
R2score01.append(R2(zreal01, zpredict301))

print ("Mean squared error (noise= 0.01, degree = 4): %.4f" % mse(zreal01, zpredict401))
print("R2 score (noise= 0.01, degree = 4): %.4f" % R2(zreal01, zpredict401))

MSE01.append(mse(zreal01, zpredict401))
R2score01.append(R2(zreal01, zpredict401))

print ("Mean squared error (noise= 0.01, degree = 5): %.4f" % mse(zreal01, zpredict501))
print("R2 score (noise= 0.01, degree = 5): %.4f" % R2(zreal01, zpredict501))

MSE01.append(mse(zreal01, zpredict501))
R2score01.append(R2(zreal01, zpredict501))


plt.plot(degree, MSE01, label="MSE")
plt.plot(degree, R2score01, label="R2 score")
plt.title('MSE and R2 score for model with noise = 0.01')
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
###############################################################################

###############################################################################
print("MSE and R2 for 0.1 noise and degree = 2, 3, 4, 5")
MSE1 = []
R2score1 = []
degree = [2, 3, 4, 5]

print ("Mean squared error (noise= 0.1, degree = 2): %.4f" % mse(zreal1, zpredict21))
print("R2 score (noise= 0.1, degree = 2): %.4f" % R2(zreal1, zpredict21))

MSE1.append(mse(zreal1, zpredict21))
R2score1.append(R2(zreal1, zpredict21))

print ("Mean squared error (noise= 0.1, degree = 3): %.4f" % mse(zreal1, zpredict31))
print("R2 score (noise= 0.1, degree = 3): %.4f" % R2(zreal1, zpredict31))

MSE1.append(mse(zreal1, zpredict31))
R2score1.append(R2(zreal1, zpredict31))

print ("Mean squared error (noise= 0.1, degree = 4): %.4f" % mse(zreal1, zpredict41))
print("R2 score (noise= 0.1, degree = 4): %.4f" % R2(zreal1, zpredict41))

MSE1.append(mse(zreal1, zpredict41))
R2score1.append(R2(zreal1, zpredict41))

print ("Mean squared error (noise= 0.1, degree = 5): %.4f" % mse(zreal1, zpredict51))
print("R2 score (noise= 0.1, degree = 5): %.4f" % R2(zreal1, zpredict51))

MSE1.append(mse(zreal1, zpredict51))
R2score1.append(R2(zreal1, zpredict51))

plt.plot(degree, MSE1, label="MSE")
plt.plot(degree, R2score1, label="R2 score")
plt.title('MSE and R2 score for model with noise = 0.1')
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
###############################################################################

###############################################################################
print("MSE and R2 for 0.2 noise and degree = 2, 3, 4, 5")
MSE2 = []
R2score2 = []
degree = [2, 3, 4, 5]

print ("Mean squared error (noise= 0.2, degree = 2): %.4f" % mse(zreal2, zpredict22))
print("R2 score (noise= 0.2, degree = 2): %.4f" % R2(zreal2, zpredict22))

MSE2.append(mse(zreal2, zpredict22))
R2score2.append(R2(zreal2, zpredict22))

print ("Mean squared error (noise= 0.2, degree = 3): %.4f" % mse(zreal2, zpredict32))
print("R2 score (noise= 0.2, degree = 3): %.4f" % R2(zreal2, zpredict32))

MSE2.append(mse(zreal2, zpredict32))
R2score2.append(R2(zreal2, zpredict32))

print ("Mean squared error (noise= 0.2, degree = 4): %.4f" % mse(zreal2, zpredict42))
print("R2 score (noise= 0.2, degree = 4): %.4f" % R2(zreal2, zpredict42))

MSE2.append(mse(zreal2, zpredict42))
R2score2.append(R2(zreal2, zpredict42))

print ("Mean squared error (noise= 0.2, degree = 5): %.4f" % mse(zreal2, zpredict52))
print("R2 score (noise= 0.2, degree = 5): %.4f" % R2(zreal2, zpredict52))

MSE2.append(mse(zreal2, zpredict52))
R2score2.append(R2(zreal2, zpredict52))

plt.plot(degree, MSE2, label="MSE")
plt.plot(degree, R2score2, label="R2 score")
plt.title('MSE and R2 score for model with noise = 0.2')
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

###############################################################################

###############################################################################
print("MSE and R2 for 0.3 noise and degree = 2, 3, 4, 5")
MSE3 = []
R2score3 = []
degree = [2, 3, 4, 5]

print ("Mean squared error (noise= 0.3, degree = 2): %.4f" % mse(zreal3, zpredict23))
print("R2 score (noise= 0.3, degree = 2): %.4f" % R2(zreal3, zpredict23))

MSE3.append(mse(zreal3, zpredict23))
R2score3.append(R2(zreal3, zpredict23))

print ("Mean squared error (noise= 0.3, degree = 3): %.4f" % mse(zreal3, zpredict33))
print("R2 score (noise= 0.3, degree = 3): %.4f" % R2(zreal3, zpredict33))

MSE3.append(mse(zreal3, zpredict33))
R2score3.append(R2(zreal3, zpredict33))

print ("Mean squared error (noise= 0.3, degree = 4): %.4f" % mse(zreal3, zpredict43))
print("R2 score (noise= 0.3, degree = 4): %.4f" % R2(zreal3, zpredict43))

MSE3.append(mse(zreal3, zpredict43))
R2score3.append(R2(zreal3, zpredict43))

print ("Mean squared error (noise= 0.3, degree = 5): %.4f" % mse(zreal3, zpredict53))
print("R2 score (noise= 0.3, degree = 5): %.4f" % R2(zreal3, zpredict53))

MSE3.append(mse(zreal3, zpredict53))
R2score3.append(R2(zreal3, zpredict53))


plt.plot(degree, MSE3, label="MSE")
plt.plot(degree, R2score3, label="R2 score")
plt.title('MSE and R2 score for model with noise = 0.3')
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
###############################################################################

###############################################################################
print("MSE and R2 for 0.4 noise and degree = 2, 3, 4, 5")
MSE4 = []
R2score4 = []
degree = [2, 3, 4, 5]

print ("Mean squared error (noise= 0.4, degree = 2): %.4f" % mse(zreal4, zpredict24))
print("R2 score (noise= 0.4, degree = 2): %.4f" % R2(zreal4, zpredict24))

MSE4.append(mse(zreal4, zpredict24))
R2score4.append(R2(zreal4, zpredict24))

print ("Mean squared error (noise= 0.4, degree = 3): %.4f" % mse(zreal4, zpredict34))
print("R2 score (noise= 0.4, degree = 3): %.4f" % R2(zreal4, zpredict34))

MSE4.append(mse(zreal4, zpredict34))
R2score4.append(R2(zreal4, zpredict34))

print ("Mean squared error (noise= 0.4, degree = 4): %.4f" % mse(zreal4, zpredict44))
print("R2 score (noise= 0.4, degree = 4): %.4f" % R2(zreal4, zpredict44))

MSE4.append(mse(zreal4, zpredict44))
R2score4.append(R2(zreal4, zpredict44))

print ("Mean squared error (noise= 0.4, degree = 5): %.4f" % mse(zreal4, zpredict54))
print("R2 score (noise= 0.4, degree = 5): %.4f" % R2(zreal4, zpredict54))

MSE4.append(mse(zreal4, zpredict54))
R2score4.append(R2(zreal4, zpredict54))

plt.plot(degree, MSE4, label="MSE")
plt.plot(degree, R2score4, label="R2 score")
plt.title('MSE and R2 score for model with noise = 0.4')
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
###############################################################################
MSEtotal = [MSE, MSE001, MSE01, MSE1, MSE2, MSE3, MSE4]
R2total = [R2score, R2score001, R2score01, R2score1, R2score2, R2score3, R2score4]