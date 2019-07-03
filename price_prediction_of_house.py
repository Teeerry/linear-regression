'''
Created on July 3, 2019

@author: Terry
@email：terryluohello@qq.com
'''
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def housePrediction():
    """ 波士顿房价预测

    描述：scikit-learn机器学习：常用算法原理及编程实战 示例5.5 测算房价
    INPUT：
        无
    OUPUT： 
        无 
    """
    # 导入数据
    boston = load_boston()
    x = boston.data
    y = boston.target
    # 拆分数据集为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    # 使用线性回归算法进行预测
    model = LinearRegression(normalize=True)
    # 计算时间
    start = time.clock()
    # 训练模型并且计算得分
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    cv_score = model.score(x_test, y_test)
    print('elaspe: {0:.6f};train_score: {1:0.6f}; cv_score: {2:.6f}'.format(time.clock()-start, train_score, cv_score))

    # 使用二阶多项式来你和数据
    model = polynomial_model(degree=2)
    # 计算时间
    start = time.clock()
    # 训练模型并且计算得分
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    cv_score = model.score(x_test, y_test)
    print('elaspe: {0:.6f};train_score: {1:0.6f}; cv_score: {2:.6f}'.format(time.clock()-start, train_score, cv_score))

def polynomial_model(degree=1):
    """ 多项式模型函数

    INPUT：
        degree：多项式的项数，默认为1
    OUPUT： 
        pipeline：多项式模型
    """
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features",polynomial_features),
                        ("linear_regression",linear_regression)])
    return pipeline

if __name__ == "__main__":
    housePrediction()