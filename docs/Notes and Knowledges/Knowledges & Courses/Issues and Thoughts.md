# Issues and Thoughts

## DQN Frame Skipping

I have trouble understanding the different with frame skipping and frame stacking, and the DeepMind paper is really not clear about it:

- [https://www.reddit.com/r/reinforcementlearning/comments/fucovf/confused_about_frame_skipping_in_dqn/](https://www.reddit.com/r/reinforcementlearning/comments/fucovf/confused_about_frame_skipping_in_dqn/)

The post also have a link to a great article which explain in depth wtf is going on with stacking and skipping:

- [https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)

## Conv layers shape of inputs

Convolution layers take their input in the following form: (width, height, channels).

```python
inputs = layers.Input(shape=(84, 84, 4))
layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
```

To pass “4 frames” to the input of a convolution layer, I basically needs to pass each frame in the “channels” parts. It does not matter because the images are greyscaled, so the channel should have been 1. So I can pass 4 images which all have 1 channel.

If I wanted to pass RGB image, I need to pass one image as (84, 84, 3), 3 is for RGB, is I want to pass one image grayscaled, it is (84, 84, 1)

### What Atari environment is returning then?

The baseline framework as well as tf_agents gym wrapper have an atari wrapper. This atari wrapper basically transform each images to grayscale and stack 4 frames onto each other. However, they give images in the shape (84, 84, 4), which means that it is composed of 84 sub-array composed of another 84 sub array and then composed of an array of 4 float. Those 4 floats represent the greyscale value for each of the **four** 84x84 images.

On the contrary, Gymnasium wrapper is FrameStack wrapper is putting the number of frame at the beginning (4, 84, 84), which means that there is 4 tables each composed of 84 sub-array composed of 84 values. 

So, I can either reshape the gymnasium wrapper to put the frame number in last to feed the Convolutional layer OR I can change the Convolutional layer by changing the data_format parameter as follows:

```python
inputs = layers.Input(shape=(4, 84, 84))
layer1 = layers.Conv2D(32, 8, strides=4, activation="relu", data_format="channel_first")(inputs)
```

It is important to pay attention to the input given to the CNN. The CNN can only take either (width, height, channels) or (channels, width, height).