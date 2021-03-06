'''
训练过程：
1.load_examples中通过tf.wholefilereader读取图像文件,tf.train.batch()获取批量样本
2.使用unet生成256*256*1图像
3.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import sys
import utils_file
import utils_image

import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export", "use", "use_batch"])
#模型和event的保存位置
parser.add_argument("--output_dir", required=False, help="where to put output files")
parser.add_argument("--seed", type=int)
#checkpoint路径
parser.add_argument("--checkpoint",  help="")
#最大训练步数
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=500, help="update summaries every summary_freq steps")
#多少步打印一次loss
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="")
parser.add_argument("--save_freq", type=int, default=500, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--start_global_step", type=int, default=0, help="restore or net")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=1.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=0.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

parser.add_argument("--image_path", default=None)
parser.add_argument("--image_list", default=None)
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256
# 定义一个namedtuple类型Examples，并包含paths，inputs和targets等属性。
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model",
                               "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")



#将图像数据范围[0,1]转为[-1,1]
def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

#将图像数据范围[-1,1]转为[0,1]
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb

#判别器的卷积层
#padded_input = [batch,258,258,4]
def discrim_conv(batch_input, out_channels, stride):
    #[1,1][1,1]对应上下左右各添加一行
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                            kernel_initializer=tf.random_normal_initializer(0, 0.02))

#生成器的卷积层
def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                kernel_initializer=initializer)

#生成器的反卷积层
def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          kernel_initializer=initializer)

#leakRelu，x>0时,return x;x<0时返回ax
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                    ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                    xyz_normalized_pixels ** (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                    fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                    (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

#源码中的结果
def load_examples1():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)#paths='merge_image\\612_250.jpg"
        raw_input = decode(contents)#raw_input.shape=(256,1024,1)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            a_images = preprocess(raw_input[:, :width // 2, :])
            b_images = preprocess(raw_input[:, width // 2:, :])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
                                                              batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg   #解码jpeg格式的图片
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))  #os.path.basename(path)返回文件名,os.path.splitext分离文件名和扩展名
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):  #isdigit()检测字符串是否只由数字组成
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")  #创建一个保持文件的FIFO队列，以供reader使用
        reader = tf.WholeFileReader()  #一个阅读器，读取整个文件，返回文件名称key,以及文件中所有的内容value
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents, 1)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)#图像归一化到[0,1]
        

        # here we horizontally expand the chanel, first three block is original image, others are mask
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 1, message="image does not have 1 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)  #返回一个一模一样新的tensor的op

        # raw_input.set_shape([None, None, 1])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            chanel = tf.split(raw_input, 4, 1)#按照宽度切成4份
            image = tf.concat(chanel[:3], 2)#image存储前三份，rgb图像合并为[256,256,3]
            masks = tf.concat(chanel[3:], 2)#[256,256,1]
            a_images = preprocess(image)
            b_images = preprocess(masks)
#            print("a_images:",a_images)
#            print("b_images:",b_images)

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)  #对图像进行从左到右翻转

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)
    #[inut_images]=[256,256,3],inuts_batch[1,256,256,3]
    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
                                                              batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size)) #向上取整

    # with  tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in range(100):
    #         temp = sess.run(raw_input)
    #         temp1 = sess.run(paths)
    #         temp2 = sess.run(input_images)
    #         temp3 = sess.run(paths_batch)
    #         temp4 = sess.run(inputs_batch)
    #         temp5 = sess.run(targets_batch)
    #         # bgr2rgb(temp4[0])
    #         # bgr2rgb(temp5[0])
    #
    #         cv2.imwrite("input_batch.bmp",(temp4[0]+1)/2*255)
    #         cv2.imwrite("targets_batch.bmp",(temp5[0]+1)/2*255)
    #         temp6,temp7,temp8,temp9 = sess.run([image,masks,inputs,targets])
    #         #temp7 = sess.run(b_images)
    #         # //bgr2rgb(temp6)
    #         # //bgr2rgb(temp7)
    #         cv2.imshow("image",temp6)
    #         cv2.imshow("mask",temp7)
    #         cv2.waitKey(0)
    #         cv2.imwrite("input.bmp",(temp6+1)/2*255)
    #         cv2.imwrite("target.bmp",(temp7+1)/2*255)
    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

#opencv存储数据为gbr，如果想用CV2保存图像，会用到该函数
def bgr2rgb(image):
    for j in range(image.shape[0]):#图像高
        for i in range(image.shape[1]):#图像宽
            image[j][i][0],image[j][i][2] = image[j][i][2],image[j][i][0]
def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]输出节点128
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]256
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]512
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]512
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]512
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]512
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]512
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)#len(layers)=8
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # [batch, height, width, 1] +[batch,height,width,3]=> [batch, height, width, 4]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]
    #unet生成图像
    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    #输入图像和目标图像合并后的卷积结果[batch,30,30,1]
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)
    # 输入图像和unet生成图像合并后的卷积结果[batch,30,30,1]
    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1，判别器用来判别真假，认为真正的输入输出对为1，生成的对为0,
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1为了骗过判别器，生成图像争取以假乱真，尽量接近1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight
    #更新判别器的训练参数
    with tf.name_scope("discriminator_train"):
        #tf.trainable_variables返回的是需要训练的变量列表
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        #计算loss中可训练的var_list中的梯度。相当于minimize()的第一步，返回(gradient, variable)对的list。
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        #minimize()的第二部分，返回一个执行梯度更新的ops。
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
    #更新生成器的训练参数
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

#对于每张测试图像，在a.output_dir+"images"目录中生成三张图像，inputs,outputs,targets
def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        # os.path.splitext()将文件名和扩展名分开，os.path.basename(),返回path最后的文件名
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    # 训练的时候的参数
    a.progress_freq =1
    a.input_dir = "./merge_image"
    a.output_dir = "./ckt"#模型和event保存目录
    a.which_direction = "AtoB"
    a.checkpoint = './ckt'
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)
   
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    #生成event和模型保存目录
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")
        
        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE        #256
        a.flip = False

    for k, v in a._get_kwargs():   #输出各个参数的值
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f: 
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))   #将参数写入options.json中

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.uint8, shape=[256, 256, 3])  #此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值

        input_image = tf.image.convert_image_dtype(input, dtype=tf.float32)  #转换为float32类型，并做归一化
        batch_input = tf.expand_dims(input_image, axis=0)  #增加1维

        with tf.variable_scope("generator"):  #tf.name_scope() 主要是用来管理命名空间的，tf.variable_scope() 的作用是为了实现变量共享
            batch_output = deprocess(create_generator(preprocess(batch_input), 1))
        #out_image.name = strided_slice:0, tf.image.convert_image_dtype返回的名称为trided_slice:0
        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        print(output_image.name)
        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key": tf.identity(key).name,
            "output": output_image.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        #tf.train.Saver给所有变量添加save和restore操作
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            #这个方法运行构造器为恢复变量所添加的操作。它需要启动图的Session。restore对变量不需要经过初始化，恢复作为初始化的一种方法。
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            #保存模型的方法:1.export_meta_grapth+saver.save(,False),分别保存图文件和参数文件;2.saver.save("",True)，同时保存图和参数文件
            #export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            #运行通过构造器添加的操作。它需要启动图的session。save要求被保存的变量必须经过了初始化。

            #export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)
            restore_saver.save(sess, os.path.join(a.output_dir, "export"))

        return
    #训练开始，1加载数据集
    examples = load_examples()
    print("examples count = %d" % examples.count)
    print(examples.inputs)
    print(examples.targets)
    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,#批处理读取的文件路径，reader.read()返回的第一个参数
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # tf.summary.scalar("discriminator_loss", model.discrim_loss)
    # tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    '''
    保存模型，
    1、创建saver时，可以指定保存的模型个数，利用max_to_keep = 4，则最终会保存4个模型
    2、saver.save,根据给定了文件名和迭代次数，生成四个文件名：如文件名+"-"+迭代次数.meta
    存储网络结构.meta、存储训练好的参数.data和.index、记录最新的模型checkpoint。
    '''
    saver = tf.train.Saver(max_to_keep=1)
   
    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    #sv = tf.train.MonitoredTrainingSwssion(checekpoint_dir = logdir)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with sv.managed_session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))
        #如果指定了checkpoint文件所在目录，且存在checkpoint文件，则加载最近的模型
        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
            if a.start_global_step > 0:
                sess.run(tf.assign(sv.global_step, a.start_global_step))
        #设置最大迭代次数
        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            #样本集中所有数量作为最大步数
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()
            
            for step in range(max_steps):
                #隔多少步返回true
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    # 配置运行时需要记录的信息
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    # 运行时记录运行信息的proto
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches
                
                results = sess.run(fetches, options=options, run_metadata=run_metadata)
                #每隔多少步更新一下日志文件
                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                
                
                
                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                
                
                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                #没隔多少步打印一次结果
                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    #steps_per_epoch:所有训练集使用一遍的次数，也就是总样本数/batch_size
                    #train_epoch:数据集的第几次epoch
                    #train_step:在第几个epoch的训练集的步数
                    #rate:1s内反向运算能够完成的图像数目
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("step: %d"%step)
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %d s" % (
                        train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])
                
                
                #每隔多少步保存一次模型,模型名称为"model"+"-"+global_step
                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


def get_mask(im, input_placeholder, output_tensor, sess, log=False):
    feed_dict = {input_placeholder: im}
    fetchs = output_tensor
    output_ = sess.run(fetchs, feed_dict)
    return output_

#获取连通区域的外界矩形，以列表形式返回
def get_cc(im_bin):
    connectivity = 8
    output = cv2.connectedComponentsWithStats(im_bin, connectivity,cv2.CV_32S)
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    #output[2]为shape为(n,5)的ndarray，5个值分别为x,y,width,height,width*height
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    max_area_index = 0
    min_thresh = 15 * 15
    cc_list = []
    #从下标1开始，0为图像的大小
    for i in range(1, len(stats)):
        if stats[i][4] > min_thresh:
            cc_list.append(stats[i][:4])#list中存放着一维的ndarray
    return cc_list


def sort_roi_lit(roi_list):
    # first sort by y
    #改为按面积排序，由大到小
    roi_list = sorted(roi_list, key=lambda x: x[2]*x[3],reverse = True )
    return roi_list[0]

    # then sort with x if distance of y is small and of x is big
    # def swap(alist, i, j):
    #     tmp = alist[i]
    #     alist[i] = alist[j]
    #     alist[j] = tmp
    #     return alist
    #
    # if roi_list[0][0] > roi_list[1][0]:
    #     roi_list = swap(roi_list, 0, 1)
    # if roi_list[4][0] > roi_list[5][0]:
    #     roi_list = swap(roi_list, 4, 5)
    # if roi_list[8][0] > roi_list[9][0]:
    #     roi_list = swap(roi_list, 8, 9)
    #
    # # roi_list = np.array(roi_list)
    # # avg_height = np.average(roi_list, axis=1)[3]
    # # for i in range(len(roi_list)):
    # #     if i > 0 and abs(roi_list[i][1] - roi_list[i - 1][1]) < avg_height / 2 and roi_list[i][0] < roi_list[i - 1][0]:
    # #         tmp = roi_list[i]
    # #         roi_list[i] = roi_list[i - 1]
    # #         roi_list[i - 1] = tmp
    # return roi_list


def get_roi_list(mask, log=False):
    ret, im_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    if log:
        utils_image.show_image(mask, 'mask', 2)
        utils_image.show_image(im_bin, 'bin', 0)
    cc_list = get_cc(im_bin)
    cc_list = sort_roi_lit(cc_list)
    return cc_list


def use(image_path, out_dir=None, log=True):
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            print("loaded", key, "=", val)
            setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    original = cv2.imread(image_path)

    with tf.Session() as sess:
        #加载模型数据
        print("loading graph")
        saver = tf.train.import_meta_graph(os.path.join(a.checkpoint, "export.meta"))
        print(sess.graph)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print("loading data")
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        saver.restore(sess, checkpoint)
        #获取mask图像
        input = sess.graph.get_tensor_by_name('Placeholder:0')
        output = sess.graph.get_tensor_by_name('strided_slice:0')#根据名称返回tensor
        width = original.shape[1]
        height = original.shape[0]
        im = cv2.resize(original, (256, 256))
        mask = get_mask(im, input, output, sess, True)
        # mask = cv2.resize(mask, (width, height))
        if log:
            cv2.imshow('src', im)
            cv2.imshow('mask', mask)
            #cv2.waitKey(0)

        roi_list = get_roi_list(mask)
        x_scale = 256 / width
        y_scale = 256 / height
        #将坐标转到原图坐标系中
        roi_list = np.array(roi_list).reshape([-1, 2]) * np.array([1 / x_scale, 1 / y_scale], dtype=np.float32)
        roi_list = roi_list.reshape([-1, 4]).astype(np.int16)
        for i, roi in enumerate(roi_list):
            #在原图中截取银行卡矩形区域
            roi_im,mask_im = utils_image.get_roi(original, roi, 'box')
            if log:
                cv2.imshow('roi', roi_im)
                cv2.imshow("mask",mask_im)
                cv2.waitKey(0)

        return


def use_batch():
    if a.image_list is None:
        raise ValueError("image_list is required...")
    if a.output_dir is None:
        raise ValueError("output_dir is required...")

    # for i in range(20):
    #     i_path = os.path.join( a.output_dir, '{:0>2}'.format(i))
    #     if os.path.exists(i_path):
    #         print(i_path, 'already exist...')
    #     else:
    #         print(i_path, 'does not exist, creating...')
    #         os.mkdir(i_path)

    # with open(os.path.join(a.checkpoint, "options.json")) as f:
    #     for key, val in json.loads(f.read()).items():
    #         print("loaded", key, "=", val)
    #         setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    image_path_list = utils_file.file_line_to_list(a.image_list)
    with tf.Session() as sess:
        print("loading graph")
        saver = tf.train.import_meta_graph(os.path.join(a.checkpoint, "export.meta"))
        print(sess.graph)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print("loading data")
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        saver.restore(sess, checkpoint)
        input = sess.graph.get_tensor_by_name('Placeholder:0')
        output = sess.graph.get_tensor_by_name('strided_slice:0')
        log = False
        debug = False
        for image_path in image_path_list:
            if not os.path.exists(image_path):
                print('file not exist: ' + image_path)
                continue
            print(image_path)
            original = cv2.imread(image_path);
            if original is None:
                print('image is None: ' + image_path)
                continue
            if debug:
                print('original.shape: ', original.shape)
            if log:
                utils_image.show_image(original, 'original', 1)

            width = original.shape[1]
            height = original.shape[0]
            im = cv2.resize(original, (256, 256))
            mask = get_mask(im, input, output, sess, False)
            name = os.path.basename(image_path)
            cv2.imwrite(os.path.join('.\\test_res',name),mask)
            # mask = cv2.resize(mask, (width, height))
            if log:
                cv2.imshow('src', im)
                cv2.imshow('mask', mask)
                cv2.waitKey(0)

            roi_list = get_roi_list(mask)
            x_scale = 256 / width
            y_scale = 256 / height
            roi_list = np.array(roi_list).reshape([-1, 2]) * np.array([1 / x_scale, 1 / y_scale], dtype=np.float32)
            roi_list = roi_list.reshape([-1, 4]).astype(np.int16)
            if len(roi_list) == 0:
                 print("none",image_path)
                 name = os.path.basename(image_path)
                 cv2.imwrite(os.path.join(a.output_dir,name))
                 continue


            for i, roi in enumerate(roi_list):
                roi_im,mask_im = utils_image.get_roi(original, roi, 'box')
                if log:
                    cv2.imshow('roi', roi_im)
                    cv2.waitKey(0)

                if a.output_dir:
                    i_dir = os.path.join(a.output_dir, '{:0>2}'.format(i))
                    mask_dir =os.path.join(a.output_dir,"mask")
                    if not os.path.exists(mask_dir):
                        os.makedirs(mask_dir)
                    name = os.path.basename(image_path)
                    i_path = os.path.join(i_dir, name)
                    mask_path = os.path.join(mask_dir,name)
                    cv2.imwrite(mask_path,mask_im)
                    if os.path.exists(i_dir):
                        cv2.imwrite(i_path, roi_im)
                    else:
                        print(i_dir, 'does not exist, creating...')
                        os.mkdir(i_dir)
                        cv2.imwrite(i_path, roi_im)

if __name__ == "__main__":
    if a.mode == 'use': #单张图像测试
        #use(a.image_path)
        #use(r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\test\src_image\630769138_350_rotate.jpg")
        #use(r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\test\src_image\630883904_350_rotate.jpg")
        #use(r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\test\src_image\631314973_10_rotate.jpg")
        #use(r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\test\src_image\631100712_270_rotate.jpg")
        #use(r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\test\src_image\631062964_350_rotate.jpg")
        #use(r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\test\src_image\631054112_350_rotate.jpg")
        use(r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\test\src_image\631062964_350_rotate.jpg")
        use(r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\test\src_image\630894943_10_rotate.jpg")
    elif a.mode == 'use_batch':#批量测试图像
        a.output_dir = "./use_batch"
        a.image_list = r"E:\project\pix2pix_idcard\pix2pix_copyto_gali\pix2pix_text\error_name\error_name/imglist_file.txt"
        use_batch()
    else:
        main()


