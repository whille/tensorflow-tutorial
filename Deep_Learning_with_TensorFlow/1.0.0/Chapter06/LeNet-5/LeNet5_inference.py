import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 6
CONV1_SIZE = 5

CONV2_DEEP = 16
CONV2_SIZE = 5

FC_SIZE = 512

def create_conv2d(input, layer_name, num_out, conv_size, stride=1, padding='SAME'):
    with tf.variable_scope(layer_name):
        W = tf.get_variable(
            "weight", [conv_size, CONV1_SIZE, input.shape[-1], num_out],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        B = tf.get_variable("bias", [num_out], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(tf.nn.bias_add(conv, B))

def create_fc(input, layer_name, num_out, regularizer=None):
    with tf.variable_scope(layer_name):
        W = tf.get_variable("weight", [input.shape[-1], num_out],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer:
            tf.add_to_collection('losses', regularizer(W))
        B = tf.get_variable("bias", [num_out], initializer=tf.constant_initializer(0.1))
    return tf.matmul(input, W) + B

def inference(input_tensor, train, regularizer):
    relu1 = create_conv2d(input_tensor, 'layer1-conv', CONV1_DEEP, CONV1_SIZE, padding='VALID')
    with tf.name_scope("layer2-pool"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1, 2, 2, 1],strides=[1, 2, 2, 1],padding="VALID")
    relu2 = create_conv2d(pool1, 'layer3-conv', CONV2_DEEP, CONV1_SIZE, padding='VALID')
    with tf.name_scope("layer4-pool"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # flat
    shape = pool2.get_shape().as_list()
    n_flat = shape[1] * shape[2] * shape[3]
    reshaped = tf.reshape(pool2, [shape[0], n_flat])

    fc1 = create_fc(reshaped, 'layer5-fc', FC_SIZE, regularizer)
    fc1 = tf.nn.relu(fc1)
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    logit = create_fc(fc1, 'layer6-fc', NUM_LABELS, regularizer)
    return logit
