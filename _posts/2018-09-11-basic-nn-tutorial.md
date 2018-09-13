---
layout: post
title: Basic Introduction to Neural Networks
tags: [ tutorial, neural_networks ]
---

A neural network is a system of neurons interconnected by nodes - by providing a certain input the system can learn to output a desired response. There are two types of problems that neural networks handle: classification and regression. In a classification problem the network aims to use provided inputs to calculate the probabilities of a certain discrete output, for example cat/dog. Regression problems are more complex, here the system aims to predict a continuous value from the inputs.

In a supervised learning problem, the neural network learns how to deal with the inputs by training on a set of training data. Using this, it aims to reduce the error by adjusting the weights of the nodes accordingly. In this tutorial we will walk through the process of predicting outcomes and adjusting the weights to improve the model. Not all concepts will be covered in depth in this tutorial.

For this tutorial, we’re going to use a neural network with two inputs, two hidden neurons, two output neurons. Additionally, the hidden and output neurons will include a bias. The architecture of the network can be seen in the image below.

![basic_nn_annotated](/images/basic_nn/tut_basic_nn_annotated.png)

In order to have some numbers to work with, here are the initial weights, the biases, and training inputs/outputs:

![basic_nn_annotated](/images/basic_nn/tut_basic_nn_numbers.png)

## The forward pass:

In the forward pass we predict the outputs based on the current architecture's weights and biases - in order to do this we need to work through the network in a systematic approach.

First we calculate the *total net input* to each hidden layer neuron, we then apply an activation function to *squash* the input - this process is repeated for each neuron upto and including the output layer.

In order to caluclate the value of h1, we need to take into the account the values of all the prior nodes which connect to it.

![basic_nn_annotated](/images/basic_nn/tut_basic_nn_calc_h1.png)

This image shows a more indepth view of how h1 will be calculated:

![basic_nn_annotated](/images/basic_nn/tut_basic_nn_calc_indepth.png)

**To Calculate h1:**

Calculating *total net input*:

<br>

$$ net_{h_{i}} = \left( \sum_{m}^{n} w_{mi} i_m \right) + b $$

$ net_{h_{1}} = w_{11}^{1} i_1 + w_{21}^{1} i_2 + b_1 $

$ net_{h_{1}} = 0.15 * 0.05 + 0.20 * 0.10 + 0.35 = 0.3775 $

<br>

Calculating *output*:

<br>

$$ out_{h_{i}} = \frac{1}{1+e^{net_{hi}}} $$

$ out_{h_{1}} = \frac{1}{1+e^{net_{h1}}} $

$ out_{h_{1}} = \frac{1}{1+e^{0.3775}} = 0.593269992 $

In order to calculate the output, we have used the 'sigmoid' activation function.

<br>

**To Calculate h2:**

Calculating *total net input*:

<br>

$ net_{h_{2}} = w_{12}^{1} i_1 + w_{22}^{1} i_2 + b_2 $

$ net_{h_{2}} = 0.25 * 0.05 + 0.30 * 0.10 + 0.35 = 0.3925 $

<br>

Calculating *output*:

<br>

$ out_{h_{2}} = \frac{1}{1+e^{net_{h2}}} $

$ out_{h_{2}} = \frac{1}{1+e^{0.3925}} = 0.596884378 $

![basic_nn_annotated](/images/basic_nn/tut_basic_nn_calc_o1.png)

**To Calculate o1:**

In order to calculate the values past the first hidden layer, we must use the values generated as outputs from the previous layer, namely: $ out_{h1} $ and $ out_{h2} $

Calculating *total net input*:

<br>

$ net_{o_{1}} = w_{11}^{2} out_{h_{1}} + w_{21}^{2} out_{h_{2}} + b_3 $

$ net_{o_{1}} = 0.40 * 0.593269992 + 0.45 * 0.596884378 + 0.35 = 1.105905967 $

<br>

Calculating *output*:

<br>

$ out_{o_{1}} = \frac{1}{1+e^{out_{o1}}} $

$ out_{o_{1}} = \frac{1}{1+e^{1.105905967}} = 0.75136507 $

<br>

**To Calculate o2:**

Calculating *total net input*:

$ net_{o_{2}} = w_{12}^{2} out_{h_{1}} + w_{22}^{2} out_{h_{2}} + b_3 $

$ net_{o_{2}} = 0.50 * 0.593269992 + 0.45 * 0.596884378 + 0.55 = 1.224921404 $

Calculating *output*:

$ out_{o_{2}} = \frac{1}{1+e^{out_{o2}}} $

$ out_{o_{2}} = \frac{1}{1+e^{1.224921404}} = 0.772928465 $

## Calculating Total Error:

In this tutorial we are going to use the 'mean-squared error' (mse) function to calculate the loss. In order to do this we calculate the loss from each output neuron and sum them:

$ E_{total} = \sum \frac {1}{2}(ideal - out)^2 $

$ E_{o1} = \frac {1}{2}(0.01 - 0.75136507)^2 = 0.274811083 $

$ E_{o2} = \frac {1}{2}(0.99 - 0.772928465)^2 = 0.023560026 $

$ E_{total} = E_{o1} + E_{o2} = 0.274811083 + 0.023560026 = 0.298371109 $

## The Backwards Pass:

The backwards pass is known as backpropogation, the aim is to update all of the weights so that they cause the actual output of the network to be closer to the target output. In order to do this we need to minimise the error of the network as a whole.

### The Output Layer:

**Updating $ {\bf w_{11}^{2}} $:**

First of all we need to figure out how much a change in $ w_{11}^{2} $ affects the total error, this is: $ \frac{\partial E_{total}}{\partial w_{11}^{2}} $ 

If we look at the diagram below it may be easier to see what is happening:

![basic_nn_annotated](/images/basic_nn/tut_basic_nn_calc_backprop_.png)

By using the chain rule:

$ \frac{\partial E_{total}}{\partial w_{11}^{2}} = \frac{\partial E_{total}}{\partial out_{o1}} \frac{\partial out_{o1}}{\partial net_{o1}} \frac{\partial net_{o1}}{\partial w_{11}^{2}} $

The red lines show the steps we make in order to get from our error to the weight.

How much does $ E_{total} $ changes based on $ out_{o1} $:

$ E_{total} = \frac {1}{2}(ideal_{o1} - out{o1})^2 + \frac {1}{2}(ideal_{o2} - out{o2})^2 $

$ \frac{\partial E_{total}}{\partial out_{o1}} = - \left( ideal_{o1} - out{o1} \right) $

$ \frac{\partial E_{total}}{\partial out_{o1}} = -(0.01 - 0.75136507) = 0.74136507 $

How much does $ out_{o1} $ changes with $ net_{o1} $:

$ out_{o_{1}} = \frac{1}{1+e^{out_{o1}}} $

$ \frac{\partial out_{o1}}{\partial net_{o1}} = out_{o1} \left( 1 - out_{o1} \right) $

$ \frac{\partial out_{o1}}{\partial net_{o1}} = 0.75136507(1 - 0.75136507) = 0.186815602 $

How much does $ net_{o1} $ change with $ w_{11}^{2} $:

$ net_{o_{1}} = w_{11}^{2} out_{h_{1}} + w_{21} out_{h_{2}} + b_3 $

$ \frac{\partial net_{o1}}{\partial w_{11}^{2}} = out_{h_{1}} $

$ \frac{\partial net_{o1}}{\partial w_{11}^{2}} = 0.593269992 $

We can finally combine all of these terms:

$ \frac{\partial E_{total}}{\partial w_{11}^{2}} = \frac{\partial E_{total}}{\partial out_{o1}} \frac{\partial out_{o1}}{\partial net_{o1}} \frac{\partial net_{o1}}{\partial w_{11}^{2}} $

$ \frac{\partial E_{total}}{\partial w_{11}^{2}} = - \left( ideal_{o1} - out{o1} \right) out_{o1} \left( 1 - out_{o1} \right) out_{h_{1}} $

$ \frac{\partial E_{total}}{\partial w_{11}^{2}} = 0.74136507 * 0.186815602 * 0.593269992 = 0.0821671041 $

In order to speed up the process we can assign:

$ \delta_{o1} = \frac{\partial E_{total}}{\partial out_{o1}} \frac{\partial out_{o1}}{\partial net_{o1}} \therefore \frac{\partial E_{total}}{\partial w_{j1}^{2}} = \delta_{o1} \frac{\partial net_{o1}}{\partial w_{j1}^{2}} $

$ \delta_{o1} = \frac{\partial E_{total}}{\partial out_{o1}} \frac{\partial out_{o1}}{\partial net_{o1}} = 0.74136507 * 0.186815602 = 0.13849856185 $

In order to decrease the error, we subtract this value, multiplied by $ \eta $ (the learning rate), from the previous weight. For the sake of this exercise we will set $ \eta $ = 0.5

$ w_{11}^{2+} = w_{11}^{2} - \eta \frac{\partial E_{total}}{\partial w_{11}^{2}} $

$ w_{11}^{2+} = 0.40 - 0.5 * 0.0821671041 = 0.35891648 $

**Updating $ {\bf w_{21}^{2}} $:**

As we now know the value of $ \delta_{o1} $ we can find $ \frac{\partial E_{total}}{\partial w_{21}^{2}} $ easily:

$ \frac{\partial E_{total}}{\partial w_{21}^{2}} = \delta_{o1} \frac{\partial net_{o1}}{\partial w_{21}^{2}} $

$ \frac{\partial E_{total}}{\partial w_{21}^{2}} = \delta_{o1} * out_{h_{2}} $

$ \frac{\partial E_{total}}{\partial w_{21}^{2}} = 0.13849856185 * 0.596884378  = 0.082667628 $

Now we have this value we can update the weight:

$ w_{21}^{2+} = w_{21}^{2} - \eta \frac{\partial E_{total}}{\partial w_{21}^{2}} $

$ w_{21}^{2+} = 0.45 - 0.5 * 0.082667628 = 0.408666186 $

**Updating $ {\bf w_{12}^{2}} $:**

Once again we need to go through the process of the chain rule, but now we know the derivatives this will be a lot easier:

$ \delta_{o2} = \frac{\partial E_{total}}{\partial out_{o2}} \frac{\partial out_{o2}}{\partial net_{o2}} $

$ \delta_{o2} = -\left(ideal_{o2} - out{o2}\right) * out_{o2} \left( 1 - out_{o2} \right) $

$ \delta_{o2} = -(0.99 - 0.772928465) * 0.772928465(1 - 0.772928465) =  -0.03809823661 $

From here we can easily calculate the new weight:

$ \frac{\partial E_{total}}{\partial w_{12}^{2}} = \delta_{o2} \frac{\partial net_{o2}}{\partial 
_{12}^{2}} $

$ \frac{\partial E_{total}}{\partial w_{12}^{2}} = \delta_{o2} out_{h_{1}} = -0.03809823661 * 0.593269992 = -0.022602541 $

$ w_{12}^{2+} = w_{12}^{2} - \eta \frac{\partial E_{total}}{\partial w_{12}^{2}} $

$ w_{12}^{2+} = 0.50 - 0.5 * -0.022602541 = 0.51130127 $

**Updating $ {\bf w_{22}^{2}} $:**

As we now know $ \delta_{o2} $ this will be very fast:

$ \frac{\partial E_{total}}{\partial w_{22}^{2}} = \delta_{o2} \frac{\partial net_{o2}}{\partial w_{22}^{2}} $

$ \frac{\partial E_{total}}{\partial w_{22}^{2}} = \delta_{o2} out_{h_{2}} = -0.03809823661 * 0.596884378 = -0.022740242 $

$ w_{22}^{2+} = w_{22}^{2} - \eta \frac{\partial E_{total}}{\partial w_{22}^{2}} $

$ w_{22}^{2+} = 0.55 - 0.5 * -0.022740242 = 0.561370121 $

### The Hidden Layer:

**Updating $ {\bf w_{11}^{1}} $:**

Calculating the updated weights in the hidden layer is a little more difficult as we need to take more factors into account - however, the approach is very much the same.

We need to figure out how much a change in $ w_{11}^{1} $ affects the total error, this is: $ \frac{\partial E_{total}}{\partial w_{11}^{1}} $ 

By using the chain rule:

$ \frac{\partial E_{total}}{\partial w_{11}^{1}} = \frac{\partial E_{total}}{\partial out_{h1}} \frac{\partial out_{h1}}{\partial net_{h1}} \frac{\partial net_{h1}}{\partial w_{11}^{1}} $

$ \frac{\partial E_{total}}{\partial w_{11}^{1}} = \left( \frac{\partial E_{o1}}{\partial out_{o1}} \frac{\partial out_{o1}}{\partial net_{o1}} \frac{\partial net_{o1}}{\partial out_{h1}} + \frac{\partial E_{o2}}{\partial out_{o2}} \frac{\partial out_{o2}}{\partial net_{o2}} \frac{\partial net_{o2}}{\partial out_{h1}} \right) $

$ \frac{\partial E_{total}}{\partial out_{h1}} = \frac{\partial E_{o1}}{\partial out_{h1}} + \frac{\partial E_{o2}}{\partial out_{h1}} $

$ \frac{\partial E_{o1}}{\partial out_{h1}} = \frac{\partial E_{o1}}{\partial out_{o1}} \frac{\partial out_{o1}}{\partial net_{o1}} \frac{\partial net_{o1}}{\partial out_{h1}} $

$ \frac{\partial E_{o1}}{\partial out_{h1}} = -\left(ideal_{o1} - out_{o1}\right) out_{o1} \left( 1 - out_{o1} \right) w_{11}^{2} $

$ \frac{\partial E_{o1}}{\partial out_{h1}} = 0.74136507 * 0.186815602 * 0.4 = 0.055399425 $

<br>

$ \frac{\partial E_{o2}}{\partial out_{h1}} = \frac{\partial E_{o2}}{\partial out_{o2}} \frac{\partial out_{o2}}{\partial net_{o2}} \frac{\partial net_{o2}}{\partial out_{h1}} $

$ \frac{\partial E_{o2}}{\partial out_{h1}} = -\left(ideal_{o2} - out_{o2}\right) out_{o2} \left( 1 - out_{o2} \right) w_{12}^{2} $

$ \frac{\partial E_{o2}}{\partial out_{h1}} = -(0.99 - 0.772928465) * 0.772928465(1 - 0.772928465) * 0.5 = -0.019049118 $

<br>

$ \frac{\partial E_{total}}{\partial out_{h1}} = 0.055399425 - 0.019049118 = 0.036350307 $

<br>

$ \frac{\partial out_{h1}}{\partial net_{h1}} = out_{h1}(1 - out_{h1}) = 0.593269992(1 - 0.593269992) = 0.241300709 $

$ \frac{\partial net_{h1}}{\partial w_{11}^{1}} = i_1 = 0.05 $

<br>

$ \frac{\partial E_{total}}{\partial w_{11}^{1}} = \frac{\partial E_{total}}{\partial out_{h1}} \frac{\partial out_{h1}}{\partial net_{h1}} \frac{\partial net_{h1}}{\partial w_{11}^{1}}  = 0.036350307 * 0.241300709 * 0.05 = 0.000438568 $

This can be written in the form:

$ \frac{\partial E_{total}}{\partial w_{11}^{1}} = \left( \sum_{o} \frac{\partial E_{total}}{\partial out_{o}} \frac{\partial out_{o}}{\partial net_{o}} \frac{\partial net_{o}}{\partial out_{h1}} \right) \frac{\partial out_{h1}}{\partial net_{h1}} \frac{\partial net_{h1}}{\partial w_{11}^{1}} $

$ \frac{\partial E_{total}}{\partial w_{11}^{1}} = \left( \sum_{o} \delta_{o} \frac{\partial net_{o}}{\partial out_{h1}} \right) out_{h1}(1 - out_{h1}) i_1 $

$ \frac{\partial E_{total}}{\partial w_{11}^{1}} = \delta_{h1} i_1 $

In this instance:

$ \delta_{h1} = 0.036350307 * 0.241300709 = 0.008771354 $

To update the weight:

$ w_{11}^{1+} = w_{11}^{1} - \eta \frac{\partial E_{total}}{\partial w_{11}^{1}} $

$ w_{11}^{1+} = 0.15 - 0.5 * 0.000438568 = 0.149780716 $

**Updating $ {\bf w_{21}^{1}} $:**

$ \frac{\partial E_{total}}{\partial w_{21}^{1}} = \delta_{h1} \frac{\partial net_{h1}}{\partial w_{21}^{1}} $

$ \frac{\partial E_{total}}{\partial w_{21}^{1}} = \delta_{h1} i_2 $

$ \frac{\partial E_{total}}{\partial w_{21}^{1}} = 0.008771354 * 0.10 = 0.0008771354 $

To update the weight:

$ w_{21}^{1+} = w_{21}^{1} - \eta \frac{\partial E_{total}}{\partial w_{21}^{1}} $

$ w_{21}^{1+} = 0.20 - 0.5 * 0.0008771354 = 0.1995614323 $

**Updating $ {\bf w_{12}^{1}} $:**

$ \frac{\partial E_{total}}{\partial w_{12}^{1}} = \frac{\partial E_{total}}{\partial out_{h2}} \frac{\partial out_{h2}}{\partial net_{h2}} \frac{\partial net_{h2}}{\partial w_{11}^{1}} $

$ \frac{\partial E_{total}}{\partial out_{h2}} = \frac{\partial E_{o1}}{\partial out_{h2}} + \frac{\partial E_{o2}}{\partial out_{h2}} $

$ \frac{\partial E_{o1}}{\partial out_{h2}} = \frac{\partial E_{o1}}{\partial out_{o1}} \frac{\partial out_{o1}}{\partial net_{o1}} \frac{\partial net_{o1}}{\partial out_{h2}} $

$ \frac{\partial E_{o1}}{\partial out_{h2}} = -\left(ideal_{o1} - out_{o1}\right) out_{o1} \left( 1 - out_{o1} \right) w_{21}^{2} $

$ \frac{\partial E_{o1}}{\partial out_{h2}} = 0.74136507 * 0.186815602 * 0.45 = 0.0623243528 $

<br>

$ \frac{\partial E_{o2}}{\partial out_{h2}} = \frac{\partial E_{o2}}{\partial out_{o2}} \frac{\partial out_{o2}}{\partial net_{o2}} \frac{\partial net_{o2}}{\partial out_{h2}} $

$ \frac{\partial E_{o2}}{\partial out_{h2}} = -\left(ideal_{o2} - out_{o2}\right) out_{o2} \left( 1 - out_{o1} \right) w_{22}^{2} $

$ \frac{\partial E_{o2}}{\partial out_{h2}} = -(0.99 - 0.772928465) * 0.772928465(1 - 0.772928465) * 0.55 = -0.02095403013 $

$ \frac{\partial E_{total}}{\partial out_{h2}} = 0.0623243528 - 0.02095403013 = 0.041370322 $

$ \frac{\partial out_{h2}}{\partial net_{h2}} = out_{h2}(1 - out_{h2}) = 0.596884378(1 - 0.596884378) = 0.2406134173 $

$ \frac{\partial net_{h2}}{\partial w_{12}^{1}} = i_1 = 0.05 $

$ \frac{\partial E_{total}}{\partial w_{12}^{1}} = \frac{\partial E_{total}}{\partial out_{h2}} \frac{\partial out_{h2}}{\partial net_{h2}} \frac{\partial net_{h2}}{\partial w_{11}^{1}} = 0.041370322 * 0.2406134173 * 0.05 = 0.0004977127 $

In this instance:

$  \delta_{h2} = 0.041370322 * 0.2406134173 = 0.00995425455 $

To update the weight:

$ w_{12}^{1+} = w_{12}^{1} - \eta \frac{\partial E_{total}}{\partial w_{12}^{1}} $

$ w_{12}^{1+} = 0.25 - 0.5 * 0.0004977127 = 0.2497511436 $

**Updating $ {\bf w_{22}^{1}} $:**

$ \frac{\partial E_{total}}{\partial w_{22}^{1}} = \delta_{h2} \frac{\partial net_{h1}}{\partial w_{22}^{1}} $

$ \frac{\partial E_{total}}{\partial w_{22}^{1}} = \delta_{h2} i_2 $

$ \frac{\partial E_{total}}{\partial w_{22}^{1}} = 0.00995425455 * 0.10 = 0.000995425455 $

To update the weight:

$ w_{22}^{1+} = w_{22}^{1} - \eta \frac{\partial E_{total}}{\partial w_{22}^{1}} $

$ w_{21}^{1+} = 0.30 - 0.5 * 0.000995425455 = 0.2995022873 $

This working constitutes one iteration, at the end of this iteration the weights are updated to their new values and the process repeats itself - working to minimise the error. I have tried to simplify the nomenclature so that you can focus on the core concepts - once you have a good understanding of this process there is a lot you can start to do to build on this knowledge. I hope you found this post useful - I will be writing about the more advanced mathematics in due time.
