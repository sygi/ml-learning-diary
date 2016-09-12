import tensorflow as tf
import math
import reading
import random

BATCH_SIZE = 20

batch = reading.get_batch(BATCH_SIZE)

n_hidden = 500
n_visible = 28 * 28

W = tf.Variable(tf.random_uniform((n_visible, n_hidden),
    minval=-4*math.sqrt(6. / (n_hidden + n_visible)),
    maxval=4*math.sqrt(6. / (n_hidden + n_visible)),
    dtype=tf.float32), name="weights")
B = tf.Variable(tf.zeros((n_hidden,), dtype=tf.float32), name="biases")

B_2 = tf.Variable(tf.zeros((n_visible,), dtype=tf.float32), name="biases_generator")
W_2 = tf.transpose(W, name="weights_generator")

sess = tf.Session()


def get_encoded(inp):
    return tf.sigmoid(tf.matmul(inp, W) + B)


def get_decoded(hidden):
    return tf.matmul(hidden, W_2) + B_2


def get_train_op(inp, noise=False):
    # TODO: add noise to inp
    z = get_encoded(inp)
    x_hat = get_decoded(z)
    loss = tf.reduce_mean(tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(x_hat, inp, name="loss"), 1), 0)

    learning_rate = 0.1  # originally: 0.1
    global_step = tf.get_variable("global_step", dtype=tf.int32,
                                  initializer=0, trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    tf.scalar_summary("loss", loss)
    x_reshaped = tf.reshape(inp, [BATCH_SIZE, 28, 28, 1])
    x_hat_reshaped = tf.reshape(tf.sigmoid(x_hat), [BATCH_SIZE, 28, 28, 1])
    tf.image_summary("input", x_reshaped)
    tf.image_summary("reconstructed", x_hat_reshaped)
    tf.histogram_summary("weights", W)

    return optimizer.minimize(loss, global_step=global_step), loss, global_step

train_op, loss_op, step_op = get_train_op(batch)
merged_op = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs/4", sess.graph, max_queue=1000)
filling_thread = reading.initialize_session(sess)

try:
    while True:
        _, l, summary, step = sess.run([train_op, loss_op, merged_op, step_op])
        writer.add_summary(summary, global_step=step)
        if random.randint(0, 1000) == 3:
            print("loss: ", l)

except tf.errors.OutOfRangeError:
    print "finished"
# TODO: some code to join the threads to be able to leave ipython
