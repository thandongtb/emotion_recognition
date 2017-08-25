import numpy as np
import tensorflow as tf
import cv2
import os.path
import imutils

class LoadTrainedModel(object):
    def __init__(self, tf_session):
        self._sess = tf_session

    def load_model(self, meta, model):
        with self._sess as sess:
            self.saver = tf.train.import_meta_graph(meta)
            self.saver.restore(sess, model)
            self.input = tf.get_collection("placeholder")
            self.output = tf.get_collection("output")

    def predict(self, image):
        h, w, d = image.shape

        if (h == w and h == 64 and d == 3):
            image_normalised = np.add(image, -127)  # normalisation of the input
            feed_dict = {self.input: image_normalised}
            roll_raw = self._sess.run([self.output], feed_dict=feed_dict)
            roll_vector = np.multiply(roll_raw, 25.0)

            return roll_vector

class LoadImage(object):
    def readImage(self, path):
        image = cv2.imread(path)
        image = imutils.resize(image, width=64)
        return image

if __name__ == '__main__':
    sess = tf.Session()
    est = LoadTrainedModel(sess)
    est.load_model('head_pose/roll/cnn_cccdd_30k.meta', 'head_pose/roll/cnn_cccdd_30k')

    image = LoadImage().readImage('tung.jpg')
    print image.shape
    print est.predict(image=image)
