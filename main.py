#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
import time

DATA_DIR = './data'
RUNS_DIR = './runs'
VGG_PATH = os.path.join(DATA_DIR, 'vgg')
NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)

## Check TensorFlow Version
#assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
#print('TensorFlow Version: {}'.format(tf.__version__))
#
## Check for a GPU
#if not tf.test.gpu_device_name():
#    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
#else:
#    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    return [graph.get_tensor_by_name(name) for name in [
        vgg_input_tensor_name,
        vgg_keep_prob_tensor_name,
        vgg_layer3_out_tensor_name,
        vgg_layer4_out_tensor_name,
        vgg_layer7_out_tensor_name]]
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    def upscale_2x(in_tensor, filters):
        """Upscales the given tensor by 2x in both width and height."""
        return tf.layers.conv2d_transpose(
            in_tensor, filters, kernel_size=4, strides=2, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    def upscale_4x(in_tensor, filters):
        """Upscales the given tensor by 4x in both width and height."""
        return tf.layers.conv2d_transpose(
            in_tensor, filters, kernel_size=16, strides=8, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    def skip_connection(layer, prev_layer):
        """Performs a skip connection by adding two non-adjacent layers."""
        return tf.add(layer, prev_layer)

    # Shape(conv_1x1): [1, 5, 18, 2], assuming IMAGE_SHAPE = (160, 576).
    conv_1x1 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, kernel_size=1, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Shape(layer): [1, 10, 36, 512].
    layer = skip_connection(upscale_2x(conv_1x1, 512), vgg_layer4_out)

    # Shape(layer): [1, 20, 72, 256].
    layer = skip_connection(upscale_2x(layer, 256), vgg_layer3_out)

    # Shape(layer): [1, 160, 576, 2].
    return upscale_4x(layer, num_classes)
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    return None, None, None
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass
#tests.test_train_nn(train_nn)


def run():
    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        vgg_tensors = [vgg_input_tensor,
                       vgg_keep_prob_tensor,
                       vgg_layer3_out_tensor,
                       vgg_layer4_out_tensor,
                       vgg_layer7_out_tensor] = load_vgg(sess, VGG_PATH)

        output = layers(vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor,
                        num_classes=NUM_CLASSES)

        #########################################################################################################
        # This section is for development only; can be removed without affecting the model.
        # Print the shapes of the tensors (need to evaluate them, since not all dimensions are known statically).
        print('Tensor shapes:')
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        tensors_to_print = [vgg_input_tensor,
                            vgg_layer3_out_tensor,
                            vgg_layer4_out_tensor,
                            vgg_layer7_out_tensor,
                            output]
        shapes = sess.run([tf.shape(t) for t in tensors_to_print],
                          # Feed dummy data.
                          {vgg_input_tensor: np.zeros([1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3]),
                           vgg_keep_prob_tensor: 1.0})
        name_to_shape = {t.name : shape for t, shape in zip(tensors_to_print, shapes)}
        print(name_to_shape)
        # Make sure the shape of the output matches the one of the input.
        in_shape = name_to_shape[vgg_input_tensor.name]
        out_shape = name_to_shape[output.name]
        np.testing.assert_array_equal(np.array(in_shape[:3]), np.array(out_shape[:3]))
        #########################################################################################################

#        # Create function to get batches
#        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), image_shape)
#
#        # OPTIONAL: Augment Images for better results
#        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
#
#        # TODO: Build NN using load_vgg, layers, and optimize function
#
#        # TODO: Train NN using the train_nn function
#
#        # TODO: Save inference data using helper.save_inference_samples
#        #  helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, image_shape, logits, keep_prob, input_image)
#
#        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
