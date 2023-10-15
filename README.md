# RNNs-implementation
Implementations of various Recurrent Neural Networks in Tensorflow 2

## Content
- `basic_builder.py`: Returns specified RNN neural network
- `custom_layers.py`: Implementation of dilated RNN layer
- `predefined_networks.py`: Examples of networks specifications based on [M4 competition winning solution by S. Smyl](https://www.sciencedirect.com/science/article/abs/pii/S0169207019301153)
- `rnn_cells.py`: Implementations of various RNN cells
- `rnn_constructor.py`: Given the RNN network specification given by either basic_builder or one of the predefined_networks this class constructs tf.keras.Model which can then be compiled and trained
- `train_predefined_networks.ipynb`: Simple example of how this repository can be used
