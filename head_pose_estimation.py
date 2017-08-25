import numpy as np
import tensorflow as tf
import cv2
import os.path

DEBUG = False


class CnnHeadPoseEstimator:
    def __init__(self, tf_session):
        self._sess = tf_session

    def print_allocated_variables(self):
        all_vars = tf.all_variables()

        print("Printing all the Allocated Tensorflow Variables:")
        for k in all_vars:
            print(k.name)     

    def _allocate_yaw_variables(self):

        self._num_labels = 1
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_yaw_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3))

        #Conv layer
        self.hy_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hy_conv1_biases = tf.Variable(tf.zeros([64]))

        #Conv layer
        self.hy_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hy_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))

        #Conv layer
        self.hy_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
        self.hy_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer
        self.hy_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1))
        self.hy_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Output layer
        self.hy_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hy_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))
 
        # Model.
        def model(data):
            X = tf.reshape(data, shape=[-1, 64, 64, 3])

            # Convolution Layer 1
            conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hy_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv1_biases))

            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)
 
            # Convolution Layer 2
            conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hy_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv2_biases))

            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hy_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv3_biases))

            # Max Pooling (down-sampling)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Apply Normalization
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Fully connected layer 4
            dense1 = tf.reshape(norm3, [-1, self.hy_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3

            dense1 = tf.tanh(tf.matmul(dense1, self.hy_dense1_weights) + self.hy_dense1_biases)

            #Output layer 6
            out = tf.tanh(tf.matmul(dense1, self.hy_out_weights) + self.hy_out_biases)

            return out
 
        # Get the result from the model
        self.cnn_yaw_output = model(self.tf_yaw_input_vector)

 
    def load_yaw_variables(self, YawFilePath):
        self._allocate_yaw_variables()

        if (os.path.isfile(YawFilePath)==False):
            raise ValueError('The yaw file path is incorrect.')

        tf.train.Saver(({"conv1_yaw_w": self.hy_conv1_weights, "conv1_yaw_b": self.hy_conv1_biases,
                         "conv2_yaw_w": self.hy_conv2_weights, "conv2_yaw_b": self.hy_conv2_biases,
                         "conv3_yaw_w": self.hy_conv3_weights, "conv3_yaw_b": self.hy_conv3_biases,
                         "dense1_yaw_w": self.hy_dense1_weights, "dense1_yaw_b": self.hy_dense1_biases,
                         "out_yaw_w": self.hy_out_weights, "out_yaw_b": self.hy_out_biases
                        })).restore(self._sess, YawFilePath) 


    def return_yaw(self, image, radians=False):
         h, w, d = image.shape

         if (h == w and h==64 and d==3):
             image_normalised = np.add(image, -127) #normalisation of the input
             feed_dict = {self.tf_yaw_input_vector : image_normalised}
             yaw_raw = self._sess.run([self.cnn_yaw_output], feed_dict=feed_dict)
             yaw_vector = np.multiply(yaw_raw, 100.0)
             #yaw = yaw_raw #* 100 #cnn out is in range [-1, +1] --> [-100, + 100]
             if (radians==True): return np.multiply(yaw_vector, np.pi/180.0) #to radians
             else: return yaw_vector

         if (h == w and h>64 and d==3):
             image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
             image_normalised = np.add(image_resized, -127) #normalisation of the input
             feed_dict = {self.tf_yaw_input_vector : image_normalised}
             yaw_raw = self._sess.run([self.cnn_yaw_output], feed_dict=feed_dict)       
             yaw_vector = np.multiply(yaw_raw, 100.0) #cnn-out is in range [-1, +1] --> [-100, + 100]
             if (radians==True): return np.multiply(yaw_vector, np.pi/180.0) #to radians
             else: return yaw_vector

         if (h != w or w<64 or h<64):
             if h != w :
                 raise ValueError('Wrong shape. Height must equal Width. Height=%d,Width=%d'%(h,w))
             else:
                 raise ValueError('Wrong shape. Height and Width must be >= 64 pixel')

         if (d!=3):
             raise ValueError('The image given as input does not have 3 channels')

    def _allocate_pitch_variables(self):
        self._num_labels = 1
        self.tf_pitch_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3))
        
        #Conv layer

        self.hp_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hp_conv1_biases = tf.Variable(tf.zeros([64]))

        #Conv layer

        self.hp_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hp_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))

        #Conv layer

        self.hp_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)) #was[3, 3, 128, 256]
        self.hp_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer

        self.hp_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1)) #was [5*5*256, 1024]
        self.hp_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Output layer

        self.hp_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hp_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))
 
        # Model.
        def model(data):

            X = tf.reshape(data, shape=[-1, 64, 64, 3])

            # Convolution Layer 1
            conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hp_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv1_biases))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)
 
            # Convolution Layer 2
            conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hp_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv2_biases))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hp_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv3_biases))
            # Max Pooling (down-sampling)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Fully connected layer 4
            dense1 = tf.reshape(norm3, [-1, self.hp_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3
            dense1 = tf.tanh(tf.matmul(dense1, self.hp_dense1_weights) + self.hp_dense1_biases)

            #Output layer 6
            out = tf.tanh(tf.matmul(dense1, self.hp_out_weights) + self.hp_out_biases)
            return out
        # Get the result from the model
        self.cnn_pitch_output = model(self.tf_pitch_input_vector)

    def load_pitch_variables(self, pitchFilePath):
        self._allocate_pitch_variables()

        if (os.path.isfile(pitchFilePath)==False): raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(load_pitch_variables): the pitch file path is incorrect.')

        tf.train.Saver(({"conv1_pitch_w": self.hp_conv1_weights, "conv1_pitch_b": self.hp_conv1_biases,
                         "conv2_pitch_w": self.hp_conv2_weights, "conv2_pitch_b": self.hp_conv2_biases,
                         "conv3_pitch_w": self.hp_conv3_weights, "conv3_pitch_b": self.hp_conv3_biases,
                         "dense1_pitch_w": self.hp_dense1_weights, "dense1_pitch_b": self.hp_dense1_biases,
                         "out_pitch_w": self.hp_out_weights, "out_pitch_b": self.hp_out_biases
                        })).restore(self._sess, pitchFilePath)

    def return_pitch(self, image, radians=False):
         h, w, d = image.shape
         #check if the image has the right shape
         if (h == w and h==64 and d==3):
             image_normalised = np.add(image, -127) #normalisation of the input
             feed_dict = {self.tf_pitch_input_vector : image_normalised}
             pitch_raw = self._sess.run([self.cnn_pitch_output], feed_dict=feed_dict)
             pitch_vector = np.multiply(pitch_raw, 45.0)
             #pitch = pitch_raw #* 40 #cnn out is in range [-1, +1] --> [-45, + 45]
             if (radians==True): return np.multiply(pitch_vector, np.pi/180.0) #to radians
             else: return pitch_vector
         #If the image is > 64 pixel then resize it
         if (h == w and h>64 and d==3):
             image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
             image_normalised = np.add(image_resized, -127) #normalisation of the input
             feed_dict = {self.tf_pitch_input_vector : image_normalised}
             pitch_raw = self._sess.run([self.cnn_pitch_output], feed_dict=feed_dict)
             pitch_vector = np.multiply(pitch_raw, 45.0) #cnn-out is in range [-1, +1] --> [-45, + 45]
             if (radians==True): return np.multiply(pitch_vector, np.pi/180.0) #to radians
             else: return pitch_vector
         #wrong shape
         if (h != w or w<64 or h<64):
             if h != w :
                 raise ValueError('Wrong shape. Height must equal Width. Height=%d,Width=%d'%(h,w))
             else:
                 raise ValueError('Wrong shape. Height and Width must be >= 64 pixel')

         if (d!=3):
             raise ValueError('The image given as input does not have 3 channels')

    def _allocate_roll_variables(self):
        self._num_labels = 1
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_roll_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3))

        #Conv layer
        self.hr_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hr_conv1_biases = tf.Variable(tf.zeros([64]))

        #Conv layer

        self.hr_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hr_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))

        #Conv layer

        self.hr_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)) #was[3, 3, 128, 256]
        self.hr_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer

        self.hr_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1)) #was [5*5*256, 1024]
        self.hr_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Output layer

        self.hr_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hr_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))
 
        # Model.
        def model(data):

            X = tf.reshape(data, shape=[-1, 64, 64, 3])

            # Convolution Layer 1
            conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hr_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),
                                           self.hr_conv1_biases))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
 
            # Convolution Layer 2
            conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hr_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hr_conv2_biases))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Convolution Layer 3
            conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hr_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hr_conv3_biases))
            # Max Pooling (down-sampling)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Fully connected layer 4
            dense1 = tf.reshape(norm3, [-1, self.hr_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3
            dense1 = tf.tanh(tf.matmul(dense1, self.hr_dense1_weights) + self.hr_dense1_biases)

            #Output layer 6
            out = tf.tanh(tf.matmul(dense1, self.hr_out_weights) + self.hr_out_biases)

            return out
 
        # Get the result from the model
        self.cnn_roll_output = model(self.tf_roll_input_vector)


    def load_roll_variables(self, rollFilePath):
        self._allocate_roll_variables()

        if (os.path.isfile(rollFilePath)==False): raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(load_roll_variables): the roll file path is incorrect.')

        tf.train.Saver(({"conv1_roll_w": self.hr_conv1_weights, "conv1_roll_b": self.hr_conv1_biases,
                         "conv2_roll_w": self.hr_conv2_weights, "conv2_roll_b": self.hr_conv2_biases,
                         "conv3_roll_w": self.hr_conv3_weights, "conv3_roll_b": self.hr_conv3_biases,
                         "dense1_roll_w": self.hr_dense1_weights, "dense1_roll_b": self.hr_dense1_biases,
                         "out_roll_w": self.hr_out_weights, "out_roll_b": self.hr_out_biases
                        })).restore(self._sess, rollFilePath)


    def return_roll(self, image, radians=False):
         h, w, d = image.shape

         if (h == w and h==64 and d==3):
             image_normalised = np.add(image, -127) #normalisation of the input
             feed_dict = {self.tf_roll_input_vector : image_normalised}
             roll_raw = self._sess.run([self.cnn_roll_output], feed_dict=feed_dict)
             roll_vector = np.multiply(roll_raw, 25.0)
             #cnn out is in range [-1, +1] --> [-25, + 25]
             if (radians==True):
                 return np.multiply(roll_vector, np.pi/180.0) #to radians
             else:
                 return roll_vector

         if (h == w and h>64 and d==3):
             image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
             image_normalised = np.add(image_resized, -127) #normalisation of the input
             feed_dict = {self.tf_roll_input_vector : image_normalised}
             roll_raw = self._sess.run([self.cnn_roll_output], feed_dict=feed_dict)       
             roll_vector = np.multiply(roll_raw, 25.0) #cnn-out is in range [-1, +1] --> [-45, + 45]
             if (radians==True): return np.multiply(roll_vector, np.pi/180.0) #to radians
             else: return roll_vector

         if (h != w or w<64 or h<64):
             if h != w :
                 raise ValueError('Wrong shape. Height must equal Width. Height=%d,Width=%d'%(h,w))
             else:
                 raise ValueError('Wrong shape. Height and Width must be >= 64 pixel')

         if (d!=3):
             raise ValueError('The image given as input does not have 3 channels')


