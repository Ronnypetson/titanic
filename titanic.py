import tensorflow as tf
import numpy as np
import csv
#	Data reading and treatment
train_path = 'train.csv'
#
reader = csv.reader(open(train_path,'rb'),delimiter=',')
train_lst = list(reader)
train_brute = np.array(train_lst)
brute_r, brute_c = train_brute.shape
#
# Survived,Pclass1,Pclass2,Pclass3,Sex,Age,SibSp,Parch,Fare,EmbarkedS,EmbarkedC,EmbarkedQ
# 1			2						4	5	6		7	9		11		
# Retirar 0 3 8 10
num_samples = brute_r-1
num_features = brute_c-1
train_data = np.zeros((num_samples,num_features))		# np.float64 is default
labels = np.zeros((num_samples,1))
#
for i in range(1,brute_r):
	col_count = 0
	row_count = i-1
	labels[row_count] = train_brute[i,1]
	for j in range(brute_c):
		if j == 2:
			# 1 = [1 0 0] 2 = [0 1 0] 3 = [0 0 1]
			if train_brute[i,j] == '1':
				train_data[row_count,col_count] = 1.0
				train_data[row_count,col_count+1] = 0.0
				train_data[row_count,col_count+2] = 0.0
			elif train_brute[i,j] == '2':
				train_data[row_count,col_count] = 0.0
				train_data[row_count,col_count+1] = 1.0
				train_data[row_count,col_count+2] = 0.0
			elif train_brute[i,j] == '3':
				train_data[row_count,col_count] = 0.0
				train_data[row_count,col_count+1] = 0.0
				train_data[row_count,col_count+2] = 1.0
			col_count += 3
		elif j == 4:
			# male = 1.0, female = 0.0
			if train_brute[i,j] == 'male':
				train_data[row_count,col_count] = 1.0
			elif train_brute[i,j] == 'female':
				train_data[row_count,col_count] = 0.0
			col_count += 1
		elif j == 5 or j == 6 or j == 7 or j == 9:
			if train_brute[i,j] != '':
				train_data[row_count,col_count] = float(train_brute[i,j])
			else:
				train_data[row_count,col_count] = 20.55
			col_count += 1
		elif j == 11:
			# S = [1 0 0] C = [0 1 0] Q = [0 0 1]
			if train_brute[i,j] == 'S':
				train_data[row_count,col_count] = 1.0
				train_data[row_count,col_count+1] = 0.0
				train_data[row_count,col_count+2] = 0.0
			elif train_brute[i,j] == 'C':
				train_data[row_count,col_count] = 0.0
				train_data[row_count,col_count+1] = 1.0
				train_data[row_count,col_count+2] = 0.0
			elif train_brute[i,j] == 'Q':
				train_data[row_count,col_count] = 0.0
				train_data[row_count,col_count+1] = 0.0
				train_data[row_count,col_count+2] = 1.0
			col_count += 3
#
print("Features: ",train_data.shape)
print("Labels: ",labels.shape)
#	Tensorflow part
# One layer regression
X = tf.placeholder(tf.float32, shape=[None,num_features])
w = tf.Variable(tf.random_normal([num_features,1],stddev=0.1))
y = tf.nn.softmax(tf.matmul(X,w))
y_ = tf.placeholder(tf.float32, [None,1])
#
cross_entropy = tf.reduce_mean(tf.abs(y-y_))
			#tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
			#-y_*tf.log(y)-(1.0-y_)*tf.log(1.0-y))
			#(y_-y)*(y_-y)
train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)
#
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#
num_epochs = 10000
bsize = 20
for i in range(num_epochs):
	j = np.random.randint(0,num_samples-bsize)
	batch_xs, batch_ys = [train_data[j:j+bsize],labels[j:j+bsize]]
	sess.run(train_step, feed_dict={X:batch_xs, y_:batch_ys})
	print(sess.run(cross_entropy, feed_dict={X:batch_xs, y_:batch_ys}))
#

