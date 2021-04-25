import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
pi = math.pi

#Barrier for Emotion Data (df_emo)
def PointsInCircum(r,n=70):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]
li = PointsInCircum(11.025)
x, y = zip(*li)
#plt.scatter(*zip(*li))
#plt.show()


X2,X1=[i for i in x if i<0 ],[j for j in x if j>0]
Y2,Y1=[i for i in y if i<0 ],[j for j in y if j>0]
list_of_tuples = list(zip(X1, X2, Y1, Y2))
df_emo = pd.DataFrame(list_of_tuples, columns = ['X1', 'X2', 'Y1', 'Y2'])
#print(df_emo)
X_e, y_e = df_emo.iloc[:16, :].values, df_emo.iloc[17:33, :].values

#print(X_e)
#print(y_e)





#6,2 interactive Data (df_62)
def PointsInCircum(r,n=943):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]
sup = PointsInCircum(150)
x, y = zip(*sup)
#plt.scatter(*zip(*sup))
#plt.show()
X2,X1=[i for i in x if i<0 ],[j for j in x if j>0]
Y2,Y1=[i for i in y if i<0 ],[j for j in y if j>0]
list_of_tuples = list(zip(X1, X2, Y1, Y2))
df_62 = pd.DataFrame(list_of_tuples, columns = ['X1', 'X2', 'Y1', 'Y2'])
#print(df_62.head())
X_w, y_w = df_62.iloc[:235, :].values, df_62.iloc[236:471, :].values

#print(len(X_w))
#print(len(y_w))


#7,1 interactive Data (df_71)
def PointsInCircum(r,n=1648):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]
sup = PointsInCircum(262.1525)
x, y = zip(*sup)
#plt.scatter(*zip(*sup))
#plt.show()
X2,X1=[i for i in x if i<0 ],[j for j in x if j>0]
Y2,Y1=[i for i in y if i<0 ],[j for j in y if j>0]
list_of_tuples = list(zip(X1, X2, Y1, Y2))
df_71 = pd.DataFrame(list_of_tuples, columns = ['X1', 'X2', 'Y1', 'Y2'])
#print(df_71.head())
X_n, y_n = df_71.iloc[:411, :].values, df_71.iloc[412:823, :].values

#print(len(X_n))
#print(len(y_n))



