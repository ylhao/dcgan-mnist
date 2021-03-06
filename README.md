# dcgan-mnist

## 生成对抗网络

Ian J. Goodfellow 等人于 2014 年在论文 Generative Adversarial Nets 中提出了一个通过对抗过程估计生成模型的新框架。框架中同时训练两个模型：一个生成模型（generative model, G），用来捕获数据分布；一个判别模型（discriminative model, D），用来估计样本来自于训练数据的概率。G 的训练过程是将 D 错误的概率最大化。可以证明在任意函数 G 和 D 的空间中，存在唯一的解决方案，使得 G 重现训练数据分布，而 D=0.5。

生成对抗网络（GAN，Generative Adversarial Networks）的基本原理很简单：假设有两个网络，生成网络 G 和判别网络 D。生成网络 G 接受一个随机的噪声 z 并生成图片，记为G(z)，判别网络 D 的作用是判别一张图片 x 是否真实，对于输入 x，D(x) 是 x 为真实图片的概率。在训练过程中， 生成器努力让生成的图片更加真实从而使得判别器无法辨别图像的真假，而 D 的目标就是尽量分辨出真实图片和生成网络 G 产出的图片，这个过程就类似于二人博弈，G 和 D 构成了一个动态的“博弈过程”。随着时间的推移，生成器和判别器在不断地进行对抗，最终两个网络达到一个动态平衡：生成器生成的图像 G(z) 接近于真实图像分布，而判别器识别不出真假图像，即 D(G(z))=0.5。最后，我们就可以得到一个生成网络 G，用来生成图片。

![](gan_theory.png)

## 损失函数分析

- 对于真实图片，D 希望为其打上标签 1
- 对于生成图片，D 希望为其打上标签 0
- 对于生成的图片，G 希望 D 打上标签 1

我们假设 y 为图片的真实标签，x 为真实图片，z 为随机噪声，G(z) 为生成器生成的图片，D(x) 为判别器对真实图片的判别结果， D(G(z)) 为判别器对生成器生成图片的判别结果。总目标函数如下：

![](object_function.png)

D 最大化该函数，G 最小化该函数，输入为 0 ~ 1 时，log 的输出为负无穷 ~ 0。
对于真实数据，D 希望为其打上标签 1，即 D 希望 D(x) 尽可能接近 1，D 希望最大化该目标函数。
对于生成器生成的数据，D 希望为其打上标签 0，即 D 希望 D(G(z)) 尽可能接近 0，D 希望最大化该目标函数。
对于生成器生成的数据，G 希望 D 为其打上标签 1，及 G 希望 D(G(z)) 尽可能接近 1，G希望最小化该目标函数。

## 生成器的实现

可以用全连层来实现生成器，但是全连层的结构比较简单，没有利用到图片的空间信息，每个像素的地位完全相同，生成质量也比较差，更好的方式是使用 Convolution Neural Network (CNN)。DCGAN 以 CNN 为主，是一种比较稳定的 GAN 的实现方式。

## Tensorflow

- tf.nn.sigmoid_cross_entropy_with_logits
- tf.maximum
- tf.layers.conv2d
- tf.contrib.layers.batch_norm
- tf.contrib.layers.flatten
- tf.layers.dense
- tf.layers.conv2d_transpose
- tf.trainable_variables
- tf.get_collection
- tf.control_dependencies
- tf.train.AdamOptimizer

## 其它知识点

- 交叉熵
- 逆卷积

## 结果

![](samples.gif)

## Loss

![](loss.png)

## 生成的图片

调用 gen.py 来生成图片。

![](gen_1.png)
![](gen_2.png)

## 参考

- 《深度有趣》
- [GAN](https://github.com/YadiraF/GAN)
- [tf.nn.sigmoid_cross_entropy_with_logits()](https://blog.csdn.net/m0_37393514/article/details/81393819)

