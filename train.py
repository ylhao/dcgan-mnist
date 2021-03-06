#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, imageio


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


batch_size = 100
z_dim = 100


OUTPUT_DIR = 'samples'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


# X 用来传入真实的图片
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='X')
# noise 传入噪声用以生成图片
noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
# 用来判断是不是训练过程
is_training = tf.placeholder(dtype=tf.bool, name='is_training')


def lrelu(x, leak=0.2):
    """
    使用 LeakyReLU
    """
    return tf.maximum(x, leak * x)


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


def discriminator(image, reuse=None, is_training=is_training):
    """
    判别器
    """
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        # 卷积、LeakyReLU
        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))
        # 卷积
        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        # Batch Normalization、LeakyReLU
        h1 = lrelu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))
        # 卷积
        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        # Batch Normalization、LeakyReLU
        h2 = lrelu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))
        # 卷积
        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        # Batch Normalization、LeakyReLU
        h3 = lrelu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        # flatten
        h4 = tf.contrib.layers.flatten(h3)
        # 输出层，是一个全连层，神经元个数为 1
        h4 = tf.layers.dense(h4, units=1)
        return tf.nn.sigmoid(h4), h4


def generator(z, is_training=is_training):
    """
    生成器
    """
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 3
        # 全连层
        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        # Batch Normalization，relu
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))
        # 逆卷积
        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        # Batch Normalization、relu
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))
        # 逆卷积
        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        # Batch Normalization、relu
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))
        # 逆卷积
        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        # Batch Normalization、relu
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        # 逆卷积
        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=1, strides=1, padding='valid', activation=tf.nn.tanh, name='g')
        return h4


# 用传入的噪声生成一张图片
g = generator(noise)
# 判别器判断一张真实的图片，d_real 是 sigmoid 值，d_real_logits 不是 sigmoid 值
d_real, d_real_logits = discriminator(X)
# 判别器判断一张生成器生成的图片，d_fake 是 sigmoid 值，d_fake_logits 不是 sigmoid 值
d_fake, d_fake_logits = discriminator(g, reuse=True)
# 划分出生成器的参数和判别器的参数
vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
# 计算 loss
# 判别器希望为真实的图片打上标签 1
loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real)))
# 判别器希望为生成的图片打上标签 0
loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake)))
# 判别器的损失值
loss_d = loss_d_real + loss_d_fake
# 生成器希望判别器为生成的图片打上标签 1
loss_g = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake)))


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)


def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    # print(images.shape)  # (100, 28, 28)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # print(img_h, img_w)  # 28, 28
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    # print(n_plots)  # 10
    m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
    # print(images.shape[1] * n_plots + n_plots + 1)  # 291
    # print(images.shape[2] * n_plots + n_plots + 1)  # 291
    # print(m)  # (291, 291)，其实就是 11 条横线和 11 条竖线，每条线的像素值为 1，横线和竖线分出了 100 个 28 * 28 的小格，对应 100 张图片
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


# 创建 session
sess = tf.Session()
# 初始化变量
sess.run(tf.global_variables_initializer())
# 生成噪声数据，服从 -1 到 1 的均匀分布
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
samples = []
loss = {'d': [], 'g': []}
# 训练 60000 步
for i in range(2000):
    # 生成了 batch_size 个 z_dim 维的噪声数据
    # 这里是生成了 100 个 100 维的向量，这些向量中的值服从 -1 到 1 的均匀分布
    n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    # 取一个 batch 的真实数据
    batch = mnist.train.next_batch(batch_size=batch_size)[0]
    # reshape
    batch = np.reshape(batch, [-1, 28, 28, 1])
    # 将数据范围从 0 到 1 变为 -1 到 1
    batch = (batch - 0.5) * 2
    # 计算判别器 loss 值和生成器 loss 值并保存到列表中
    d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch, noise: n, is_training: True})
    loss['d'].append(d_ls)
    loss['g'].append(g_ls)
    """
    训练判别器相对来说要容易一些，训练生成器相对来说要难一些，所以多训练一次生成器，这种做法比较粗糙

    To avoid the fast convergence of D (discriminator) network,
    G (generator) network is updated twice for each D network update, which differs from original paper.
    """
    sess.run(optimizer_d, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
    # 每训练 1000 步，做一些操作
    if i % 100 == 0:
        # 打印当前步数，判别器的 loss 值和生成器的 loss 值
        print(i, d_ls, g_ls)
        # 生成图片
        gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})
        # 将数值范围从 -1 到 1 变为 0 到 1，gen_imgs 一共包含 100 张图片的信息
        gen_imgs = (gen_imgs + 1) / 2
        # print(gen_imgs)
        # 得到图片并显示图片
        # 得到 100 张 28 * 28 * 1 的图片，imgs 是一个列表，列表中是 100 个数组，每个数组是一个二维数组，shape 是 (28, 28)
        imgs = [img[:, :, 0] for img in gen_imgs]
        # print(imgs)
        gen_imgs = montage(imgs)
        # 关闭网格线
        plt.axis('off')
        plt.imshow(gen_imgs, cmap='gray')
        # 保存图片
        plt.savefig(os.path.join(OUTPUT_DIR, 'sample_{}.jpg'.format(i)))
        # 显示图片
        # plt.show()
        # 将图片添加到列表中，生成动图用
        samples.append(gen_imgs)
# 画 loss 曲线
plt.figure(figsize=(8, 6))
plt.plot(loss['d'], label='Discriminator')
plt.plot(loss['g'], label='Generator')
# 图例位于右上角
plt.legend(loc='upper right')
plt.savefig('loss.png')
# plt.show()
# 制作动图
imageio.mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=5)


# 保存模型
saver = tf.train.Saver()
saver.save(sess, './mnist_dcgan', global_step=2000)

