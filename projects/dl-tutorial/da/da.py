import tensorflow as tf
import math
import reading
import random


batch = reading.get_batch()  # need to create batch before starting queue runners

n_hidden = 500
n_visible = 28 * 28

W = tf.Variable(tf.random_uniform((n_visible, n_hidden),
    minval=-4*math.sqrt(6. / (n_hidden + n_visible)),
    maxval=4*math.sqrt(6. / (n_hidden + n_visible)),
    dtype=tf.float32), name="weights")
B = tf.Variable(tf.zeros((n_hidden,), dtype=tf.float32), name="biases")

B_2 = tf.Variable(tf.zeros((n_visible,), dtype=tf.float32), name="biases_generator")
W_2 = tf.transpose(W, name="weights_generator")

sess = tf.InteractiveSession()


def get_encoded(inp):
    return tf.sigmoid(tf.matmul(inp, W) + B)


def get_decoded(hidden):
    return tf.sigmoid(tf.matmul(hidden, W_2) + B_2)


def get_train_op(inp):
    # TODO: add noise to inp
    z = get_encoded(inp)
    x_hat = get_decoded(z)
    loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(inp, x_hat, name="loss"))
    
    learning_rate = 0.3
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    loss_summary_op = tf.scalar_summary("loss", loss)
    x_reshaped = tf.reshape(inp, [128, 28, 28, 1])
    x_hat_reshaped = tf.reshape(x_hat, [128, 28, 28, 1])
    inp_summary_op = tf.image_summary("input", x_reshaped)
    im_summary_op = tf.image_summary("reconstructed", x_hat_reshaped)

    return optimizer.minimize(loss), loss

train_op, loss_op = get_train_op(batch)
merged_op = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph)
reading.initialize_session(sess)

while True:
    _, l, summary = sess.run([train_op, loss_op, merged_op])
    if random.randint(0, 10) == 3:
        writer.add_summary(summary)
        print("loss: ", l)
        writer.flush()

# TODO: some code to join the threads to be able to leave ipython
