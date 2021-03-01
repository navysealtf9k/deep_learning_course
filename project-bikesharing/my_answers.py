import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))

        # self.bias_input_to_hidden = np.zeros((1, self.hidden_nodes))
        # self.bias_hidden_to_output = np.zeros((1, self.output_nodes))

        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

        self.activation_function_derivative = (lambda x: x * (1 - x))


    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        # delta_bias_i_h = np.zeros(self.bias_input_to_hidden.shape)
        # delta_bias_h_o = np.zeros(self.bias_hidden_to_output.shape)

        for X, y in zip(features, targets):

            # # Convert inputs into 2D numpy array
            # X = X.reshape(1, features.shape[1])

            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)

            # Implement the backproagation function below
            # delta_weights_i_h, delta_weights_h_o, delta_bias_i_h, delta_bias_h_o = self.backpropagation(final_outputs,
            #                                                                                             hidden_outputs,
            #                                                                                             X,
            #                                                                                             y,
            #                                                                                             delta_weights_i_h,
            #                                                                                             delta_weights_h_o,
            #                                                                                             delta_bias_i_h,
            #                                                                                             delta_bias_h_o)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs,
                                                                        X, y, delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        # self.update_weights(delta_weights_i_h, delta_weights_h_o, delta_bias_i_h, delta_bias_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here.

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) #+ self.bias_input_to_hidden
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) #+ self.bias_hidden_to_output
        final_outputs = final_inputs # signals from final output layer

        return final_outputs, hidden_outputs

    # def backpropagation(self, final_outputs, hidden_outputs, X, y, 
    #                     delta_weights_i_h, delta_weights_h_o, delta_bias_i_h, delta_bias_h_o):
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        # TODO: Output error - Replace this value with your calculations.
        layer_2_error = (y - final_outputs) # Output layer error is the diff between desired target and actual output.

        # TODO: Calculate the hidden layer's contribution to the error
        layer_2_delta = layer_2_error * 1

        layer_1_error = np.matmul(layer_2_delta, self.weights_hidden_to_output.T)
        layer_1_delta = layer_1_error * (self.activation_function_derivative(hidden_outputs))

        # Weight step (hidden to output)
        # delta_weights_h_o += layer_2_delta * hidden_outputs[:, None]
        delta_weights_h_o += np.multiply(layer_2_delta, hidden_outputs[:, None])
        # delta_bias_h_o += layer_2_delta

        # Weight step (input to hidden)
        delta_weights_i_h += layer_1_delta * X[:, None]
        # delta_bias_i_h += layer_1_delta

        # return delta_weights_i_h, delta_weights_h_o, delta_bias_i_h, delta_bias_h_o
        return delta_weights_i_h, delta_weights_h_o

    # def update_weights(self, delta_weights_i_h, delta_weights_h_o, delta_bias_i_h, delta_bias_h_o, n_records):
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # update weights with gradient descent step
        self.weights_hidden_to_output += (self.lr * delta_weights_h_o) / n_records 
        self.weights_input_to_hidden += (self.lr * delta_weights_i_h) / n_records 
        # self.bias_hidden_to_output += (self.lr * delta_bias_h_o) / n_records
        # self.bias_input_to_hidden += (self.lr * delta_bias_i_h) / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        # signals into hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)# + self.bias_input_to_hidden[0]
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with the appropriate calculations.
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)# + self.bias_hidden_to_output[0]
        final_outputs = final_inputs # signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 3000
learning_rate = 0.9
hidden_nodes = 10
output_nodes = 1
