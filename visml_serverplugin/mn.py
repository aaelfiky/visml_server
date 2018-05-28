import tensorflow as tf
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import StringIO

from werkzeug.routing import BaseConverter
from PIL import Image
from flask import send_file




mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_train_images_set = mnist.train.images[:50000]
mnist_train_labels_set = mnist.train.labels[:50000]
mnist_val_images_set = mnist.train.images[len(mnist_train_images_set):len(mnist.train.images)]
mnist_val_labels_set = mnist.train.labels[len(mnist_train_labels_set):len(mnist.train.labels)]
mnist_train_size = len(mnist_train_images_set)

with open('/home/ahmed/studentProject/workspace/MNIST_BK/visml_serverplugin/visml_serverplugin/predictions_labels.csv', 'rb') as f:
  reader = csv.reader(f)
  train_data_list = list(reader)
  train_data = np.array(train_data_list)

test_data_labels = train_data[:, 0]
test_data_np = train_data[:, 1:]

n_nodes_hlayer_1 = 400
# n_nodes_hlayer_1 = 500
n_nodes_hlayer_2 = 400
n_nodes_hlayer_3 = 400

n_classes = 10
batch_size = 100
trained = False

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

first_image = test_data_np[0]
first_image = np.array(first_image, dtype='float')
# first_image = train_data_np
# print(len(first_image))

# pixels = first_image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()
first_image_2 = test_data_np[1]
first_image_2 = np.array(first_image_2, dtype='float')
first_image_3 = test_data_np[2]
print(test_data_labels[2])
first_image_3 = np.array(first_image_3, dtype='float')
first_image_4 = test_data_np[3]
print(test_data_labels[3])
first_image_4 = np.array(first_image_4, dtype='float')


def gen_image(arr):
    print(len(arr))
    print(arr[0],type(arr[0]))
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img


master = Image.new(mode='RGB', size=(64, 64), color=(0, 0, 0))  # fully transparent

master.paste(gen_image(first_image), (0 * 24, 0 * 24))
master.paste(gen_image(first_image_2), (1 * 24, 0 * 24))
master.paste(gen_image(first_image_3), (0 * 24, 1 * 24))
master.paste(gen_image(first_image_4), (1 * 24, 1 * 24))


# plt.imshow(master)
# plt.show()


def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  temp = x - np.max(x)
  e_x = np.exp(x - np.max(x))
  result = e_x / float(e_x.sum())
  return result


def probabilities(x):
  min_val = np.min(x)
  negative = False
  if min_val < 0:
    negative = True

  if negative:
    new_array = x
    for i in range(0, len(x)):
      new_array[i] = x[i] + (min_val * (-1))
    x = new_array

  result = x / float(x.sum())
  return result


def neural_network_model(data):
  hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hlayer_1])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hlayer_1]))}
  hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hlayer_1, n_nodes_hlayer_2])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hlayer_2]))}
  hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hlayer_2, n_nodes_hlayer_3])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hlayer_3]))}
  output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hlayer_3, n_classes])),
                  'biases': tf.Variable(tf.random_normal([n_classes]))}

  l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
  l1 = tf.nn.relu(l1)

  l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
  l2 = tf.nn.relu(l2)

  l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
  l3 = tf.nn.relu(l3)

  output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'], name="output")

  return output


def get_next_batch(data, b, batch_size):
  index = b * batch_size
  if index + batch_size > len(data):
    return data[index:]
  else:
    return data[index:(index + batch_size)]


# t_data = [1,2,3,4,5,6,7,8,9,10]
# print(t_data[9])
# print(get_next_batch(t_data,4,2))
# prediction = object()

def train_neural_network():
  prediction = neural_network_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y), name="cost")
  optimizer = tf.train.AdamOptimizer().minimize(cost)
  hm_epochs = 10

  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # if not (trained):
    for epoch in range(hm_epochs):
      epoch_loss = 0
      for count in range(int(mnist_train_size / batch_size)):
        # epoch_x, epoch_y = mnist.train.next_batch(batch_size)
        epoch_x = get_next_batch(mnist_train_images_set, count, batch_size)
        epoch_y = get_next_batch(mnist_train_labels_set, count, batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
        epoch_loss += c
      print('Epoch', epoch, ' completed out of', hm_epochs, 'loss:', epoch_loss)
    # number of correct predictions in one hot encoded labels
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # print(prediction)

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    # print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    print('Accuracy: ', accuracy.eval({x: mnist_val_images_set, y: mnist_val_labels_set}))
    trained = True
    tf.add_to_collection('vars', prediction)
    saver.save(sess, './my-model')

  # 	# first_image = mnist.test.images[i]
  # 	# first_image = np.array(first_image, dtype='float')
  # 	# pixels = first_image.reshape((28, 28))
  # 	# plt.imshow(pixels, cmap='gray')
  # 	# plt.show()


def generate_predictions():
  with tf.Session() as sess:
    # 	sess.run(tf.global_variables_initializer())
    # 	# for i in range(0, len(mnist.test.images)):
    saver = tf.train.import_meta_graph('./my-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    # prediction = tf.get_collection('vars')
    prediction = graph.get_tensor_by_name("output:0")
    probs_predict = []
    one_hot_predict = []

    for i in range(0, len(test_data_np)):
      feed_dict = {x: [mnist.test.images[i]]}
      classification = prediction.eval(feed_dict=feed_dict, session=sess)

      prob = probabilities(classification)

      one = softmax(classification)
      # first_image = mnist.test.images[i]
      # first_image = np.array(first_image, dtype='float')
      # pixels = first_image.reshape((28, 28))
      # plt.imshow(pixels, cmap='gray')
      # plt.show()
      print("probabilities", prob.flatten())
      one_flat = one.flatten()
      one_flat = [np.argmax(one_flat)]
      one_flat.append(int(test_data_labels[i]))
      one_hot_predict.append(one_flat)
      print("one_flat", one_flat)
      probs_predict.append(prob.flatten())
    n_a = np.array(probs_predict)
    one_hot = np.array(one_hot_predict)
    np.savetxt("prob_predictions.csv", n_a, delimiter=",")
    np.savetxt("predictions_labels.csv", one_hot, delimiter=",")


def test_neural_network(start, end):
  # len(mnist.test.images)
  with tf.Session() as sess:
    # 	sess.run(tf.global_variables_initializer())
    # 	# for i in range(0, len(mnist.test.images)):
    saver = tf.train.import_meta_graph('./my-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    # prediction = tf.get_collection('vars')
    prediction = graph.get_tensor_by_name("output:0")
    for i in range(start, end):
      feed_dict = {x: [mnist.test.images[i]]}
      # classification = sess.run(model, feed_dict)
      # prob = tf.nn.softmax(prediction)
      # classification = trained_nn.eval(feed_dict=feed_dict, session=sess)
      classification = prediction.eval(feed_dict=feed_dict, session=sess)
      # prob = probabilities(classification)
      prob = softmax(classification)
      print("probabilities", prob)
      print("correct label", mnist.test.labels[i])


train_neural_network()

# test_neural_network(0,50)

generate_predictions()


