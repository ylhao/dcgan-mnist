# dcgan-mnist

## 生成对抗网络

Ian J. Goodfellow 等人于 2014 年在论文 Generative Adversarial Nets 中提出了一个通过对抗过程估计生成模型的新框架。框架中同时训练两个模型：一个生成模型（generative model, G），用来捕获数据分布；一个判别模型（discriminative model, D），用来估计样本来自于训练数据的概率。G 的训练过程是将 D 错误的概率最大化。可以证明在任意函数 G 和 D 的空间中，存在唯一的解决方案，使得 G 重现训练数据分布，而 D=0.5。

生成对抗网络（GAN，Generative Adversarial Networks）的基本原理很简单：假设有两个网络，生成网络 G 和判别网络 D。生成网络 G 接受一个随机的噪声 z 并生成图片，记为G(z)，判别网络 D 的作用是判别一张图片 x 是否真实，对于输入 x，D(x) 是 x 为真实图片的概率。在训练过程中， 生成器努力让生成的图片更加真实从而使得判别器无法辨别图像的真假，而 D 的目标就是尽量分辨出真实图片和生成网络 G 产出的图片，这个过程就类似于二人博弈，G 和 D 构成了一个动态的“博弈过程”。随着时间的推移，生成器和判别器在不断地进行对抗，最终两个网络达到一个动态平衡：生成器生成的图像 G(z) 接近于真实图像分布，而判别器识别不出真假图像，即 D(G(z))=0.5。最后，我们就可以得到一个生成网络 G，用来生成图片。

## 损失函数分析

判别器 D 的目的是：
- 对于真实图片，D 要为其打上标签 1
- 对于生成图片，D 要为其打上标签 0

生成器的目的是：
- 对于生成的图片，G 希望 D 打上标签 1

我们假设 y 为图片的真实标签，x 为真实图片，z 为随机噪声，G(z) 为生成器生成的图片，D(x) 为判别器对真实图片的判别结果， D(G(z)) 为判别器对生成器生成图片的判别结果。则判别器的损失函数可定义为：

```
-((1 - y) * log(1 - D(G(z))) + y * log(D(x)))
```

当输入的图片为真实图片时，判别器的损失函数可简化成 -log(D(x))，D(x) 越大，损失函数的值也越小。而我们希望的正是让 D(x) 尽量靠近 1。

当输入的图片为生成器生成的图片时，判别器的损失函数可以简化成 -log(1-D(G(z)))，D(G(z)) 越小，损失函数的值越小，我们希望的也正是让 D(x) 尽量靠近 0。

（需要继续完善这一部分）

## Tensorflow

- sigmoid_cross_entropy_with_logits
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
