###############################################################################
# Caleydo - Visualization for Molecular Biology - http://caleydo.org
# Copyright (c) The Caleydo Team. All rights reserved.
# Licensed under the new BSD license, available at http://caleydo.org/license
###############################################################################

from phovea_server.ns import Namespace
from phovea_server.util import jsonify
import logging
from PIL import Image, ImageOps
import StringIO
from flask import request
from flask import send_file
import os
import os.path
import math
import csv
import numpy as np

import flask as fl


app = Namespace(__name__)
_log = logging.getLogger(__name__)
cwd = os.path.dirname(os.path.realpath(__file__))
# root = os.path.join(cwd, '../MNIST_BK')
dbpath = os.path.join(cwd, 'mnist_test.csv')

with open(dbpath, 'rb') as f:
    reader = csv.reader(f)
    train_data_list = list(reader)
    train_data = np.array(train_data_list)
test_data_labels = train_data[:, 0]
test_data_entries = np.array(train_data[:, 1:], dtype=float)
l_entries = len(test_data_entries)


dbpath_2 = os.path.join(cwd, 'prob_predictions.csv')

with open(dbpath_2, 'rb') as f:
    reader = csv.reader(f)
    probs_list = list(reader)
    probs_array = np.array(probs_list)
prob_final = np.array(probs_array,dtype=float)
finala = []
for x in probs_array:
  temp = np.array(x,dtype=float)
  finala.append(temp)


dbpath_3 = os.path.join(cwd, 'tsne.csv')

with open(dbpath_3, 'rb') as f:
    reader = csv.reader(f)
    tsne_list = list(reader)
    tsne_array = np.array(tsne_list)
tsne_final = np.array(tsne_array,dtype=float)

dbpath_4 = os.path.join(cwd, 'predictions_labels.csv')

with open(dbpath_4, 'rb') as f:
    reader = csv.reader(f)
    pred_label = list(reader)
    pred_label_array = np.array(pred_label)
pl_final = np.array(pred_label_array,dtype=float)


def gen_image(arr):

    two_d = (np.reshape(arr, (28, 28))).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img

def serve_pil_image(pil_img):
    img_io = StringIO.StringIO()
    pil_img.save(img_io, 'PNG', quality=90)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')



@app.route('/<fromTo>')
def get_labels_predict(fromTo):
    img_ids = map(int, fromTo.split(','))
    from_ = img_ids[0]
    to = img_ids[1]
    dbpath = os.path.join(cwd, 'predictions_labels.csv')
    with open(dbpath, 'rb') as f:
            reader = csv.reader(f)
            predict_list = list(reader)
            predict_list_np = np.array(predict_list)
    test_entries_shown = predict_list_np[from_:to, :]
    return jsonify({
      'message': 'Hello World',
      'list': test_entries_shown
    })


@app.route('/tsne/', methods=['GET'])
def get_tsne_array():
  return jsonify({
    'message': 'TSNE',
    'list': tsne_final,
    'pl': pl_final,
    'prob': finala
  })



@app.route('/box/<index>', methods=['GET'])
def _get_image_sprite_box(index):

    return serve_pil_image(gen_image(test_data_entries[int(index)]))


@app.route('/sprites/', methods=['GET'])
def _get_image_sprite_boxes():
    n_w = 1
    n_h = int(math.ceil(1 / float(n_w)))

    width = n_w * 28
    height = 28

    images = []
    for i in range(0,20):
        images.append(serve_pil_image(gen_image(test_data_entries[i])))
    # return serve_pil_image(master)
    return jsonify({
      'message': 'Shater',
      'images': images
    })
# @app.route('/image/<fromN>+<to>', methods=['GET'])


@app.route('/image/<fromTo>', methods=['GET'])
def _get_image_sprite(fromTo):

    img_ids = map(int, fromTo.split(','))
    from_ = img_ids[0]
    to = img_ids[1]
    if from_ < 0:
        return jsonify({
          'message': 'ERROR'
        })
    _log.info('route 3: ' + str(l_entries) + ", " + str(to))
    if to > l_entries:
        return jsonify({
          'message': 'ERROR'
        })


    _log.info('route 2: ' + str(from_) + ", " + str(to))
    n_imgs = to - from_

    if len(img_ids) == 0:
        return 'error'

    n_w = 10
    n_h = int(math.ceil(n_imgs / float(n_w)))

    width = n_w * 52
    height = n_h * 52
    #
    master = Image.new(mode='RGB', size=(width, height), color=(0, 0, 0))  # fully transparents
    # for i in range
    test_entries_shown = test_data_entries[from_:to, :]

    k = len(test_entries_shown)
    start = 0
    # for input in test_entries_shown:
    for i in range(0, n_h):
        for j in range(0, n_w):
            if start == k:
                break

            b_img = ImageOps.expand(gen_image(test_entries_shown[start]),border = 2,fill='gold')
            master.paste(b_img, (j * 53, i * 53))
            start += 1

    return serve_pil_image(master)

@app.route('/predict/<index>', methods=['GET'])
def get_probs(index):
    return jsonify({
          'message': 'Probabilities',
          'index': index,
          'list': finala[int(index)]
        })

def create():
    return app
