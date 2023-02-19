# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import random
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from sklearn import preprocessing

# 固定随机种子
np.random.seed(10)



# 定义网络模型
def create_model(n_layers,regularizer,input_dim):

    model = Sequential()
    print(n_layers,regularizer)
    for i in range(1,round(n_layers)+1):
        model.add(Dense(round(128), input_dim=input_dim, activation='relu', kernel_regularizer=l2(regularizer)))
    model.add(Dense(1, activation='linear', kernel_regularizer=l2(0.000001)))
    model.add(Dropout(0.3))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
    return model

# 定义适应度函数
def fitness_function(params):

    data = pd.read_csv(r"data.csv",header=1)
    data = data.values
    chara = len(data[0])
    xData = data[:, 0:chara-1]
    yData = data[:, chara-1:]
    #2.归一化
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=True)
    X = scaler.fit_transform(xData)
    Y = scaler.fit_transform(yData)

    # 3.划分数据
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1, train_size=0.9,test_size=0.1)

    model = create_model(params[0],params[1],len(xData[0]))
    history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    epochs=1000,
                    batch_size=64,
                    verbose=0)
###对测试集的误差作图
    yPred = model.predict(x_test)
#model.save(r'E:\学术论文\爆炸压力统计\图片\finalModel\dnnModel.h5')

    yTestOri = pow(scaler.inverse_transform(y_test),10)#输出测试值的反归一化
    yPredOri = pow(scaler.inverse_transform(yPred.reshape(-1,1)),10)#输出预测值的反归一化
    abstError=yPredOri-yTestOri
    relError=abstError/yTestOri

#######用这个作为优化的评价函数
    meanRealError = np.mean(abs(relError))
    fitness_value = meanRealError
    return fitness_value

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # 粒子位置
        self.velocity_i=[]          # 粒子速度
        self.pos_best_i=[]          # 最佳位置
        self.err_best_i=-1          # 最佳误差
        self.err_i=-1               # 误差

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,10))
            self.position_i.append(x0[i])

    # 计算当前适应度函数
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # 检查当前适应度函数是否为最佳
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # 更新粒子速度
    def update_velocity(self,pos_best_g):
        w=0.7    # 恒定的更改速度（惯性变量）,可修改尝试
        c1=2.1        # 常数2.1
        c2=2.1        # 常数2.1

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # 更新粒子位置
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # 判断是否达到最大边界
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # 判断是否小于最小边界
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        err_best_g=-1                   # 最佳误差组
        pos_best_g=[]                   # 最佳误差组对应的粒子位置组

        # 建立粒子群
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # 循环优化
        i=0
        while i < maxiter:
           
           # 计算适应度函数
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                #确定当前是否为全局最佳
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # 循环更新粒子群
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # 最终结果
        print ('最佳参数:')
        print ('最佳参数',pos_best_g)
        print ('最小误差',err_best_g)


# 初始化粒子群位置
initial=[random.randint(3, 4),random.uniform(0.000001,0.001)]
# 参数边界
bounds=[(3,7),(0.000001,0.001)]

# 将最佳参数值

PSO(fitness_function,initial,bounds,num_particles=50,maxiter=100)





