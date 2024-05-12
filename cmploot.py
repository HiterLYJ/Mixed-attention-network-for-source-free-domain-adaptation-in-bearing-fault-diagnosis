# confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
#类别名，名字可以根据你的项目任意更改
classes = ['0', '1', '2', '3', '4','5']
#每个类别预测结果，以行的方式输入。我这里的数字代表每类的图片数
confusion_matrix = np.array(
    [(26, 1, 0, 0, 0, 2),
     (6, 15, 0, 0, 0, 0),
     (0, 0, 88, 0, 0, 34),
     (0, 0, 0, 329, 0, 0),
     (0, 0, 0,  0, 26, 0),
     (0, 0, 8, 0, 0, 53)])
#这是一个Recall的计算结果矩阵，这里我用recall_matric来做我右边colorbar的衡量指标
recall_matrix = np.array([(0.87, 0.03, 0.00, 0.00, 0.00, 0.07),
						  (0.27, 0.68, 0.00, 0.00, 0.00, 0.00),
						  (0.00, 0.00, 0.72, 0.00, 0.00, 0.28),
						  (0.00, 0.00, 0.00, 1.00, 0.00, 0.00),
						  (0.00, 0.00, 0.00, 0.00, 1.00, 0.00),
					      (0.00, 0.00, 0.13, 0.00, 0.00, 0.87)], dtype=np.float64)
# interpolation='nearest'插值方式
# cmap=plt.cm.Blues换矩阵的颜色，我记得有一张很长的表，都是关于颜色的，在CSDN可以找到哦
plt.imshow(recall_matrix, interpolation='nearest', cmap=plt.cm.Blues)
#命名标题
plt.title('Apple')
#画右边的colorbar
plt.colorbar()
tick_marks = np.arange(len(classes))
#rotation这个参数是可以控制坐标的字体的旋转方向的
plt.xticks(tick_marks, classes, rotation=-90)
plt.yticks(tick_marks, classes)
thresh = confusion_matrix.max() / 2.
iters = np.reshape([[[i, j] for j in range(6)] for i in range(6)], (confusion_matrix.size, 2))
for i, j in iters:
	 # horizontalalignment="center"解决数字不居中的问题
    plt.text(j, i, format(confusion_matrix[i, j]), horizontalalignment="center")
plt.ylabel('Ground truth')
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()
