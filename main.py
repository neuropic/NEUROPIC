
from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask, render_template, request , url_for , session , current_app

import tensorflow as tf
import tensorflow_hub as hub

import IPython.display as display               
from werkzeug.utils import secure_filename
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
from PIL import Image
import secrets

app = Flask(__name__)


def save_image(photo):
  photo_name = "neuropic.jpg"
  file_path = os.path.join(current_app.root_path,"static/images/",photo_name)
  photo.save(file_path)
  return photo_name  


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


@app.route("/")
def home():
    r = render_template('home.html')
    return r


@app.route('/recommend' , methods = ['GET', 'POST'])
def recommend():
    post = Post
    if request.method == 'POST':
      content = request.files["cimage"]
      content_file = "static/images/" +  str(secure_filename(content.filename))
      content.save(content_file)
      photoname = "neuropic.jpg"
      if photoname in os.listdir("static/images/"):
        os.remove(os.path.join(current_app.root_path,"static/images/",photoname))
        
        style_path= "images.jpg"
        content_path = content_file


        content_image = load_img(content_path)
        style_image = load_img(style_path)

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    
        cimage = save_image(tensor_to_image(stylized_image))
        return  render_template("recommend.html" , cimage = cimage ,oima = content_path )
      else:
        style_path= "images.jpg"
        content_path = content_file


        content_image = load_img(content_path)
        style_image = load_img(style_path)

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    
        cimage = save_image(tensor_to_image(stylized_image))
        
        return  render_template("recommend.html" , cimage = cimage ,oima = content_path )
  

if __name__ == '__main__':
    app.run(debug=True)