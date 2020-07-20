

import tensorflow as tf
import input_data
import matplotlib

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x=tf.placeholder(tf.float32,[None,28,28,1])
t=tf.placeholder(tf.float32,[None,10])

x=tf.reshape(x,[-1,784])


w =tf.Variable(tf.zeros(784,10))
b = tf.Variable(tf.zeros(10))

init_op=tf.global_variables_initializer()

y=tf.nn.softmax(tf.matmul(x,w)+b)

cross_entropy=-tf.reduce_sum(t*tf.log(y))

is_correct=tf.equal(tf.argmax(y,1),tf.argmax(t,1))
accuracy=tf.reduce_sum(tf.cast(is_correct,tf.float32))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.03)
train_step=optimizer.minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(init_op)


with tf.Session() as sess:
    for step in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch_xs,t: batch_ys})

        if step%100==0:
            acc,loss=sess.run([accuracy,cross_entropy])
            feed_dict={x:batch_xs,t:batch_ys}
            acc, loss = sess.run([accuracy, cross_entropy])
            feed_dict = {x: mnist.test.images, t: mnist.test.labels}

