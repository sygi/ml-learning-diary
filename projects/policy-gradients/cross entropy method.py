
# coding: utf-8

# In[1]:

from __future__ import print_function

def tf_print(tensor):
  def print_tensor(x):
    print(x)
    return x
  print(tensor.name, ":", end=' ')
  log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])
  with tf.control_dependencies(log_op):
    return tf.identity(tensor)


# In[2]:

import tensorflow as tf
import numpy as np
import gym


# In[3]:

def get_action():
  #  type: () -> (tf.python.framework.ops.Tensor, tf.python.framework.ops.Tensor, tf.python.framework.ops.Tensor)
  obs = tf.placeholder(tf.float32, shape=(5))
  params = tf.placeholder(tf.float32, shape=(5))
  action = tf.cast(tf.reduce_sum(tf.mul(params, obs)) > 0., tf.int32)
  return action, obs, params


# In[4]:

def evaluate(params_ev, env, action, obs_place, params_place, sess, render=False):
  env.reset()
  assert(env.action_space.n == 2)
  first_action = env.action_space.sample()
  obs, rew, done, _ = env.step(first_action)
  obs = np.append(obs, 1.)
  reward = rew
  it = 0
  while not done:
    if reward >= 1000:
      break
    if render:
      env.render()
    action_eval = sess.run(action, feed_dict={obs_place: obs, params_place: params_ev})
    obs, rew, done, _ = env.step(action_eval)
    obs = np.append(obs, 1.)
    if render:
      print(obs, rew, done)
    reward += rew
  env.close()
  return np.append(params_ev, reward), reward


# In[5]:

def sample_policy_params(mu, sigma):
  size = mu.get_shape()
  assert size == sigma.get_shape()
  return tf.random_normal(size, mu, stddev=(sigma + 1e-10), dtype=tf.float32)


# In[6]:

def update_parameters(mu, sigma, best_params):
  new_mu = tf.reduce_mean(best_params, 0)
  mu_ass = tf.assign(mu, new_mu)
  diff = tf.squared_difference(best_params, new_mu)
  std = tf.sqrt(tf.reduce_mean(diff, 0))
  sigma_ass = tf.assign(sigma, std)
  return mu_ass, sigma_ass


# In[7]:

def run_and_update(env, eval_results, params, action, params_place, obs_place, update_op, res_place, i_place, sess, n, size=5):
  sum_reward = 0.
  for i in range(n):
    params_ev = sess.run(params)
    res, rew = evaluate(params_ev, env, action, obs_place, params_place, sess)
    sum_reward += rew
    sess.run(update_op, feed_dict={res_place: res, i_place: i})
    print('.', end='') 

  print(sum_reward)


# In[8]:

def get_update_op(eval_results, size=5):
  res_place = tf.placeholder(tf.float32, shape=(size+1))
  i_place = tf.placeholder(tf.int32, shape=())
  update_op = tf.scatter_update(eval_results, [i_place], tf.cast(tf.reshape(res_place, [1, size+1]), tf.float32))
  return update_op, res_place, i_place


# In[9]:

def get_best_params(eval_results, k, size=5):
  eval_trans = tf.transpose(eval_results)
  best = tf.nn.top_k(eval_trans, k=k)
  best_ind = best.indices[size]
  best_params = tf.gather(eval_results, best_ind)
  return tf.slice(best_params, [0, 0], [k, size])


# In[10]:

def get_session():
  config = tf.ConfigProto(operation_timeout_in_ms=10000,
                          log_device_placement=True)
  return tf.Session(config=config)


# In[13]:

env = gym.make("CartPole-v1")
n=40
p=20
rounds=10
size=5

with get_session() as sess:
  mu = tf.Variable([0.] * 5, dtype=tf.float32, name="mu")
  sigma = tf.Variable([100.] * 5, dtype=tf.float32, name="sigma")
  eval_results = tf.Variable(np.zeros([n, 5+1]), dtype=tf.float32, name="eval_res")
  sess.run(tf.initialize_all_variables())
  policy_choice = sample_policy_params(mu, sigma)
  best_params_op = get_best_params(eval_results, (p*n)/100)
  mu_ass_op, sigma_ass_op = update_parameters(mu, sigma, best_params_op)
  action, obs_place, params_place = get_action()
  update_op, res_place, i_place = get_update_op(eval_results)
  
  for i in range(rounds):
    run_and_update(env, eval_results, policy_choice, action, obs_place, params_place, update_op, res_place, i_place, sess, n)
    print(sess.run([mu_ass_op, sigma_ass_op]))
    if i == 0:
      writer = tf.train.SummaryWriter("/tmp/cem2", sess.graph)

  params_ev = sess.run(policy_choice)
  env.monitor.start('/tmp/cem-vid',force=True)
  res = evaluate(params_ev, env, action, obs_place, params_place, sess, True)
  env.close()
  env.render()
  print(res[1])
  
env.monitor.close()
env.render(close=True)

