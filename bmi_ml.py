
# coding: utf-8

# In[81]:

import pandas as pd
import numpy as np
import tensorflow as tf


# In[82]:

csv = pd.read_csv('bmi.csv')
# csv


# In[83]:

#正規化 1未満の数に直す なんでいるんだろう?
# これで、CSVの列に一括で処理していることになる　forループ回す必要がない

csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100


# In[84]:

# ラベルを3次元で表すための辞書 なんで3次元化がいるんだろう?
bmi_dic = {'thin':[1,0,0], 'normal':[0,1,0], 'fat':[0,0,1]}


# In[85]:

# csv['label']の値に対して、lambda関数を適用して、それをcsv['label_pat']に保持する
# np.arrayはndarray型を生成し、普通の多重リストではなく、型が一定の行列でなければならず、より高速に処理される
csv['label_pat'] = csv['label'].apply(lambda x: np.array(bmi_dic[x]))
# csv


# In[86]:

# テストデータをわける
test_csv = csv[15000:20000]

#　テストのパータンデータ これでweightとheightの列を結合したCSVデータになってる!
test_pat = test_csv[['weight', 'height']]

test_ans = list(test_csv['label_pat'])

# test_csv
# test_pat
# test_ans


# In[87]:

# データフローグラフの作成
# まずプレースホルダ
x = tf.placeholder(tf.float32, [None, 2]) # 身長体重のデータ
y_ = tf.placeholder(tf.float32, [None, 3]) # 答えのラベル(値)を格納する


# In[88]:

# 変数
W = tf.Variable(tf.zeros([2, 3])) # 重み
b = tf.Variable(tf.zeros([3])) # バイアス


# In[89]:

# 変数とプレースホルダを、ソフトマックス回帰にかける
# 最大値を1に近づけ、最小値を0に近づける関数?
# http://developabout0309.blogspot.jp/2016/07/tensorflow-8.html
y = tf.nn.softmax(tf.matmul(x, W) + b)
# このyが予想値


# In[90]:

# 訓練

# 予想の値がy、正解がy_
# cross_entropy / 交差エントロピーは、小さいほど正確な値ということらしい つまりlossが少ない、ということ
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# GradientDescentOptimizerの引数は学習係数で、モデルによって最適な値が違う
# http://qiita.com/isaac-otao/items/6d44fdc0cfc8fed53657
# 実際にも調整しながら進めことが多いらしい
optimizer = tf.train.GradientDescentOptimizer(0.062)
train = optimizer.minimize(cross_entropy)


# In[91]:

# 正解率
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))


# In[92]:

# セッション開始
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #初期化


# In[93]:

#　学習させる
for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1 + i : 1 + i  +100]
    x_pat = rows[['weight', 'height']]
    y_ans = list(rows['label_pat'])
    fd = {x: x_pat, y_: y_ans}
    sess.run(train, feed_dict = fd)
    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict = fd)
        acc = sess.run(accuracy, feed_dict = fd)
        print('step = ', step, 'cre = ', cre, 'acc = ', acc)

acc = sess.run(accuracy, feed_dict = {x: test_pat, y_: test_ans})
print('正解率', acc)
