import scipy.io
data = scipy.io.loadmat('./data/ConnecticutGearData/DataForClassification_TimeDomain.mat') # 读取mat文件
print(data.keys())  # 查看mat文件中的所有变量
print(data['AccTimeDomain'])
# print(data['matrix2'])
# matrix1 = data['matrix1']
# matrix2 = data['matrix2']
# print(matrix1)
# print(matrix2)
# scipy.io.savemat('matData2.mat',{'matrix1':matrix1, 'matrix2':matrix2}) # 写入mat文件