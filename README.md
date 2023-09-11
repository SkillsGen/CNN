# CNN

This is a basic neural network library running on the cpu. It mostly
exists as a learning exercise and perhaps an example for anyone who
wants to see a very straightforward implementation of something that
is often an opaque if powerful library (or stack of libraries).


Defining the neural network will change as more features are added
(recurrance, convolution, etc) but the basic principle is to define
the structure of the NN as follows:

```
  {NEURONS, ACTIVATIONFUNCTION},
  {OUTPUTNEURONS, ACTIVATIONFUNCTION},
```


#### For example:

    cnn_u64 Layers[][2] =
    {
	      {32, CNN_ACT_RELU},
	      {32, CNN_ACT_RELU},
	      {Outputs, CNN_ACT_SIGMOID},
    };

#### Activation functions supported:<br>
    CNN_ACT_RELU,<br>
    CNN_ACT_SIGMOID,<br>
    CNN_ACT_TANH,<br>

#### Loss functions supported:<br>
    CNN_LOSS_MEANSQ,<br>
    //TODO: CNN_LOSS_CROSSENTROPY,<br>


#### Weights are initialized as follows:<br>
     RELU:           HE initialization<br>
     TANH & SIGMOID: Normalized Xavier<br>
     //NOTE: this can be customized in cnn_init_nn_rand, it's interesting how<br>
     //      fragile it can be with poor weight initialization<br>


#### Telemetry:
     By training with the TELEMETRYFLAG_ON, you will store the loss, gradients that
     have gone to zero and gradients that have exploded for each batch trained.
     The flag TELEMETRYFLAG_VALIDATEGRADIENTS will calculate the relative error of
     each gradient by calculating the finite difference of that weight and performing
     the following:<br>
           AbsoluteValue(gradient - finite_Difference)/Max(gradient,finite_Difference)
     Each batch's telemetry frame with then store the Largest error and it's position
     in the network as well as the average error.<br>
     The flag TELEMETRYFLAG_STOREGRADERRORS will store an entire copy of the NN with
     the calculated relative errors for each weight within each telemetry frame. This
     can be accessed by the same way as the values in any NN.<br>
       Because the above can take up such a huge amount of space and processing power,
     it's intended use is to train the full NN on a small portion of your dataset for
     very few batches. You can validate the structure of your NN with this telemetry and
     then continue.


#### Example Programs:

gate_example.c<br>
This is self contained

mnist_example.c<br>
The dataset to be used with this example program can be found here:<br>
http://yann.lecun.com/exdb/mnist/

ionosphere_example.c<br>
The dataset to be used with this example program can be found here:<br>
https://archive.ics.uci.edu/dataset/52/ionosphere
