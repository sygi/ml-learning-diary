import cPickle
import gzip
import numpy
import threading
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_images(im):
    plt.imshow(1 - im.reshape(28, 28), cmap=plt.cm.gray)

data_file = gzip.open("mnist.pkl.gz")
(train_np, _), (valid_np, _), (test_np, _) = cPickle.load(data_file)  # removing labels
data_file.close()

fifo_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[784])
# subs = train_np[:10000, :]
enqueue_op = fifo_queue.enqueue_many(train_np)

def fill_fifo_queue(sess):  # TODO: substitute with QueueRunner somehow
    NUM_EPOCHS = 15
    for i in range(NUM_EPOCHS):
        print("epoch ", i)
        sess.run(enqueue_op)
    sess.run(fifo_queue.close())


def get_batch(batch_size=20):
    example = fifo_queue.dequeue()
    batch = tf.train.shuffle_batch([example], batch_size=batch_size, capacity=12800, min_after_dequeue=500, seed=None, enqueue_many=False)
    return batch


def initialize_session(sess):
    tf.train.start_queue_runners(sess=sess)
    enqueue_thread = threading.Thread(target=fill_fifo_queue, args=(sess,))
    enqueue_thread.start()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    return enqueue_thread
