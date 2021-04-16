import numpy as np
import math

class Neuron:

    def __init__(self, weights, activation_f):
        self.weights = weights
        self.activation_f = activation_f
        self.last_delta = 0
        self.last_excited = 0
        self.last_activation = 0

    def adjustment(self, input, learning_rate):
        adjustment = learning_rate * self.last_delta
        delta_w = adjustment * np.array(input)
        self.weights += delta_w

    def get_activation(self, input):
        excited_state = np.inner(input, self.weights)
        self.last_excited = excited_state
        self.last_activation = self.activation_f(excited_state)
        return self.last_activation

class MultilayerPerceptron:
    
    LIMIT = 0.1
    
    def __init__(self, training_set, expected_output, learning_rate, hidden_layers):
        
        self.training_set = np.array(list(map(lambda t: [1]+t, training_set)))   # add e0 = 1
        self.expected_output = expected_output
        #TODO deshardcodear el len del expected output que esta en 1, el problema es que lo pasamos a array en vez de array de arrays entonces len(expected_output[0]) da error
        self.layers = [len(self.training_set[0])] + list(map(lambda units: units+1, hidden_layers)) + [1]     # add one unit to each layer --> v0
        self.layers_amount = len(self.layers)
        self.learning_rate = learning_rate
        self.neurons = {}
        self.neurons[0] = [Neuron(None, self.activation_function) for i in range(self.layers[0])]
        # Dictionary { key:layer value:list de activation states }

        # connect hidden_layer i con i+1
        for layer_i in range(1,self.layers_amount): #initialize hidden layers weights
            self.neurons[layer_i] = []
            for i in range(self.layers[layer_i]):
                # El nodo umbral tienen siempre activacion 1 y no tiene pesos entrantes
                if i == 0 and layer_i != self.layers_amount-1:
                    self.neurons[layer_i].append(Neuron(None,self.activation_function))
                    self.neurons[layer_i][i].last_activation = 1
                else:
                    w = np.random.rand(self.layers[layer_i-1])
                    self.neurons[layer_i].append(Neuron(w,self.activation_function)) 
        
    def activation_function(self, excited_state):
        return np.tanh(excited_state)

    def deriv_activation_function(self, h):
        return (1 - math.tanh(h)**2)

    def train(self, epochs_amount):
        error = float('inf') 
        for epoch in range(epochs_amount):
            aux_training = self.training_set.copy() 
            while error > self.LIMIT and len(aux_training) > 0: 
                i_x = np.random.randint(0, len(aux_training))                  # agarro un input random 
                self.apply_input_to_layer_zero(aux_training[i_x])              # V0_k = Eu_k
                aux_training = np.delete(aux_training, i_x, axis=0)
                
                self.propagate()

                # Calcular delta para neuronas den la ultima capa
                last_layer_neurons = self.neurons[self.layers_amount - 1]
                for i in range(self.layers[-1]):
                    last_layer_neurons[i].last_delta = self.deriv_activation_function(last_layer_neurons[i].last_excited)*(self.expected_output[i_x] - last_layer_neurons[i].last_activation)
                
                self.backpropagate()                                           # retropropagar 

                self.update_weigths()                                          # a cada neurona le actualizo el peso

                error = self.calculate_error() 



    # Cuando ya agarré el input, le asigno cada componente a el estado de activación de cada unidad de la capa cero 
    def apply_input_to_layer_zero(self, input): # input = [0 1 1 1 0]
        for i in range(len(input)):
            self.neurons[0][i].last_activation = input[i]
    
    # Calculo el estado de activación para todas las neuronas de la red para el input que agarré
    def propagate(self): 
        
        # Empiezo desde la primera capa oculta
        for layer_i in range(1,self.layers_amount): #V[m][i] = activations[layer][unit]

            neurons = self.neurons[layer_i]
            prev_layer_neurons = self.neurons[layer_i - 1]
            pln_activations = np.array(list(map(lambda n: n.last_activation, prev_layer_neurons)))

            for i in range(self.layers[layer_i]):
                # No propago en la neurona umbral
                if not (i == 0 and layer_i != self.layers_amount-1):
                    neurons[i].get_activation(pln_activations)
                    
    

    # Hago el camino inverso para calcular los deltas
    def backpropagate(self): 

        #   Para calcular el delta de un nodo, necesitamos hacer la sumatoria de los weights que salen hacia un nodo * el delta de dicho nodo
        #   Empiezo desde la primera capa oculta
        for layer_i in range(self.layers_amount - 1,1,-1): # delta[layer][unit] voy desde M-1 hasta 2     e0 -- w --- V0
            neurons = self.neurons[layer_i-1]
            upper_level_neurons = self.neurons[layer_i]
            # Si estoy en una capa oculta, saco el nodo umbral
            if layer_i != self.layers_amount - 1:
                upper_level_neurons = self.neurons[layer_i][1:]
            upper_deltas = [n.last_delta for n in upper_level_neurons]
            # No propago hacia atras sobre la neurona umbral --> solo tomo las neuronas no umbral de las capas ocultas
            for unit in range(1,self.layers[layer_i-1]):
                deriv = self.deriv_activation_function(neurons[unit].last_excited)
                w = [n.weights[unit] for n in upper_level_neurons]
                inner = np.inner(w, upper_deltas)
                neurons[unit].last_delta = deriv * inner   # delta = g' * (expected - real) se calcula el delta de la capa de salida

    def update_weigths(self): 
        for layer_i in range(1,self.layers_amount):
            # next_deltas = np.array(self.deltas[layer])
            neurons = self.neurons[layer_i]
            prev_layer_neurons = self.neurons[layer_i-1]
            pln_activations = np.array(list(map(lambda n: n.last_activation, prev_layer_neurons)))
            for unit in range(self.layers[layer_i]):
                # Las neuronas umbral no tiene pesos
                if not (unit == 0 and layer_i != self.layers_amount-1): 
                    neurons[unit].adjustment(pln_activations,self.learning_rate)


    def calculate_error(self): 
        error = 0
        for i in range(len(self.training_set)):
            expected = self.expected_output[i]
            aggregate = 0
            for j in range(self.layers[-1]):
                aggregate += self.neurons[self.layers_amount-1][j].last_excited
                # excited_state = np.inner(self.weights[self.hidden_layers_amount],self.activations[self.hidden_layers_amount])
            activation_state = self.activation_function(aggregate)
            error += (expected - activation_state)**2
        return 0.5 * error


    def get_output(self, input):
        aux_input = np.array(list(map(lambda t: [1]+t, input)))
        output = []
        for elem in aux_input:
            print(elem)
            self.apply_input_to_layer_zero(elem)
            self.propagate()
            output.append([neuron.last_activation for neuron in self.neurons[self.layers_amount-1]])

        return output
