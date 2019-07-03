'''
Created on July 3, 2019

@author: Terry
@email：terryluohello@qq.com
'''
import regression
from numpy import *

def abaloneTest(): 
    """ 预测鲍鱼的年龄

    描述：机器学习实战示例8.3 预测鲍鱼的年龄
    INPUT：
        无
    OUPUT： 
        无 
    """
    # 加载数据
    abX, abY = regression.loadDataSet("./data/abalone.txt") 
    # 使用不同的核进行预测
    oldyHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1) 
    oldyHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1) 
    oldyHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10) 
    # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    print("old yHat01 error Size is :" , regression.rssError(abY[0:99], oldyHat01.T))
    print( "old yHat1 error Size is :" , regression.rssError(abY[0:99], oldyHat1.T))
    print( "old yHat10 error Size is :" , regression.rssError(abY[0:99], oldyHat10.T)) 
    # 打印出不同的核预测值与新数据集（测试数据集）上的真实值之间的误差大小
    newyHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1) 
    print("new yHat01 error Size is :" , regression.rssError(abY[0:99], newyHat01.T))  
    newyHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1) 
    print("new yHat1 error Size is :" , regression.rssError(abY[0:99], newyHat1.T))  
    newyHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10) 
    print("new yHat10 error Size is :" , regression.rssError(abY[0:99], newyHat10.T)) 
    # 使用简单的线性回归进行预测，与上面的计算进行比较
    standWs = regression.standRegres(abX[0:99], abY[0:99]) 
    standyHat = mat(abX[100:199]) * standWs 
    print("standRegress error Size is:", regression.rssError(abY[100:199], standyHat.T.A))

if __name__ == "__main__":
    abaloneTest()