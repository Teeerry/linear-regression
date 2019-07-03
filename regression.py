'''
Created on July 3, 2019

@author: Terry
@email：terryluohello@qq.com
'''
from numpy import *

def loadDataSet(filename):
    """ 加载数据

    描述：解析以tab键分割的文件中的浮点数
    INPUT：
        filename：文件名
    OUTPUT:
        dataMat: feature 对应的数据集
        labelMat: feature 对应的分类标签(类别标签)
    """
    # 获取样本特征总数，不算最后的目标变量
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        # 读取每一行
        lineArr = []
        # 删除一行中以tab分隔的数据前后的空白符号
        curLine = line.strip().split('\t')
        # i 从0到2，不包括2
        for i in range(numFeat):
            # 将数据添加到lineArr List中，每一行测试数据组成一个行向量
            lineArr.append(float(curLine[i]))
        # 将测试数据的输入数据部分储存到dataMat的List中
        dataMat.append(lineArr)
        # 将每一行的最后一个数据，即类别，储存到labelMat List中
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat



def standRegres(xArr, yArr):
    """ 计算线性回归的回归系数w的最优解

    INPUT：
        xArr：输入的样本数据，包含每个样本的feature
        yArr：输入样本的类别标签
    OUTPUT：
        ws：回归系数
    """
    # mat函数将输入转换为矩阵
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    # 矩阵行列式不为零，则矩阵可逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 损失函数为均方误差(平方损失)
    # 根据公式西瓜书3.11，求得w的最优解
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k = 1.0):
    """ 局部加权线性回归(Locally Weighted Linear Regression)

    描述：在待测点附近的每个点赋予一定的权重，在子集上基于最小
        均方误差来进行普通的回归。
    INPUT：
        testPoint：样本点(行向量)
        xArr：输入的样本数据，包含每个样本的feature
        yArr：输入样本的类别标签
        k：关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
    OUTPUT：
        testPoint * ws：数据点与具有权重的系数相乘得到的预测点
    NOTES：
        计算权重的公式，w = e^((x^((i))-x) / -2k^2) 
        理解：x为某个预测点，x^((i))为样本点，样本点距离预测点越近，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）。
        关于预测点的选取，在我的代码中取的是样本点。其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差。
        算法思路：假设预测点取样本点中的第i个样本点（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
        也就可以计算出每个样本贡献误差的权值，可以看出w是一个有m个元素的向量（写成对角阵形式）。
    """
    # mat函数将输入转换为矩阵
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 获得xMat的行数
    m = shape(xMat)[0]
    # 创建权重矩阵weights，初始化权重为1
    # eye()返回一个对角线为1的二维数组
    weights = mat(eye(m))

    # 计算权重矩阵weights
    # 计算公式参考机器学习实战(中文版)P142页
    for j in range(m):                      
        diffMat = testPoint - xMat[j,:]   
        # k控制衰减速率  
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    
    xTx = xMat.T * (weights * xMat)
    # 矩阵行列式不为零，则矩阵可逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 计算出回归系数的一个估计
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):
    """ 测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
    
    INPUT：
        testArr：测试所用的所有样本点
        xArr：样本的特征数据，即 feature 
        yArr：每个样本对应的类别标签，即目标变量
        k：控制核函数的衰减速率
    OUPUT：
        yHat：预测点的估计值
    """
    # 样本总数
    m = shape(testArr)[0]
    yHat = zeros(m)
    # 局部加权线性回归
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0): 
    """ 首先将 X 排序，其余的都与lwlrTest相同，这样更容易绘图
    INPUT：
        xArr：样本的特征数据，即 feature 
        yArr：每个样本对应的类别标签，即目标变量，实际值
        k：控制核函数的衰减速率的有关参数，这里设定的是常量值 1 
    OUPUT：
        yHat：样本点的估计值
        xCopy：xArr的复制
    """
    # 生成一个与目标变量数目相同的 0 向量
    yHat = zeros(shape(yArr)) 
    # 将 xArr 转换为 矩阵形式
    xCopy = mat(xArr) 
    # 排序
    xCopy.sort(0) 
    # 开始循环，为每个样本点进行局部加权线性回归，得到最终的目标变量估计值
    for i in range(shape(xArr)[0]): 
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k) 
    return yHat,xCopy

def rssError(yArr,yHatArr): 
    """ 计算预测误差大小(均方误差)

    INPUT：
        yArr：实际值(数组)
        yHatArr：预测值(数组)
    OUTPUT：
        ((yArr-yHatArr)**2).sum()：均方误差
    """
    return ((yArr-yHatArr)**2).sum()
