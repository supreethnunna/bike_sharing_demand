import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights using random intialization
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # We are using sigmoid function as your activation function for the hidden layer
        
        def sigmoid(x):
            return (1 / (1 + np.exp(-x)))  
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        
        # Intially create a matrix of zeroes for the hidden inputs weights and the hidden output weights
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        
        # Neural Network Forward Pass
        
        # Hidden Layer 
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output Layer
        # We are not using any sigmoid activation function for the output as we have to predict a continous variable
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
       
        # Neural Network Backward Pass

        # Calculate the error of the neural net 
        error = y - final_outputs 
        
        # Backpropagated error terms is same as the error
        output_error_term = error
        
         # hidden layer's error
        hidden_error = np.dot(self.weights_hidden_to_output,error)
        
        hidden_error_term = hidden_error*hidden_outputs*(1.0 - hidden_outputs)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term*X[:,None]
        
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term*hidden_outputs[:,None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        
        # Gradient Descent step to update the weights of the hidden input and hidden output layers 
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records 
        
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records

    def run(self, features):
        
        # Hidden layer 
        hidden_inputs = np.dot(features,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # Output layer 
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


###################
# Set your hyperparameters here
####################
iterations = 1000
learning_rate = 0.1
hidden_nodes = 9
output_nodes = 1
