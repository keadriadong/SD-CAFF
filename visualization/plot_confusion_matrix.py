import pickle
import os
import glob
import numpy as np

'''
matrix = [[179,   9,  12,  29],
         [ 18,  89,   0,   1],
         [  8,   2,  78,  24],
         [ 13,   1,  13,  97]]

name = 'AFF_CRNN'
matrix = [[923, 61, 21, 94],
 [102, 484, 5, 17],
 [100, 0, 384,  94],
 [165,  11,  44, 443]]

name = 'CNN'
matrix = [[887,  74,  36, 102],
 [120, 469,   3,  16],
 [104,   4, 382,  88],
 [149,  17,  60, 437]]


name = 'CRNN'
matrix = [[910,  61,  25, 103],
 [115, 469,   2,  22],
 [ 98,   6, 360, 114],
 [161,  12,  46, 444]]
'''

name = 'RNN'
matrix = np.array([[852,  97,  36, 114],
 [129, 466,   6,   7],
 [108,   4, 338, 128],
 [213,   9,  67, 374]], dtype=int)

per_matrix = []
num_class = np.sum(matrix, axis=1).reshape(1,-1).tolist()[0]
print(num_class)
for i in range(len(matrix)):
    per_matrix.append([x/num_class[i] for x in matrix[i]])

print(per_matrix)
percentage = per_matrix
acc_class = []

print(percentage)
for i in range(len(percentage)):
    acc_class.append(percentage[i][i])
# print(os.path.basename(m), ': ', np.matrix(np.array(percentage)))
# print(os.path.basename(m), '_acc_class:', acc_class)


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#
font_times = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
sns.set()
plt.rc('font',family='Times New Roman')

f, ax = plt.subplots()

a = sns.heatmap(percentage, annot=True, ax=ax, cmap="Blues", fmt='.3f',
                xticklabels=['Neutral', 'Sad', 'Angry', 'Happy'], yticklabels=['Neutral', 'Sad', 'Angry', 'Happy'],
                annot_kws={'family':'Times New Roman','weight':'normal','size': 14})

'''
a = sns.heatmap(matrix, annot=True, ax=ax, cmap="GnBu", fmt='d',
                annot_kws={'family':'Times New Roman','weight':'normal','size': 14})
'''

plt.tick_params(labelsize=12)
labels = ax.set_xticklabels(('Neutral', 'Sad', 'Angry', 'Happy')) + ax.set_yticklabels(('Neutral', 'Sad', 'Angry', 'Happy'))
[label.set_fontname('Times New Roman') for label in labels]

# ax.set_title('confusion matrix')
ax.set_xlabel('Predict', font_times) #x
ax.set_ylabel('True', font_times) #y
plt.savefig('{}.png'.format(name),dpi=600)
plt.show()


