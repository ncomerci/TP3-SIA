#  V0_k  --> estado de activación de la neurona k de la capa 0 
#  w[i][j] --> peso de la neurona i de la capa siguiente con el input j
#  W[i][j] --> peso de la neurona i de la capa siguiente con el estado activación de la neurona de la capa anterior
import numpy as np
class MultilayerPerceptron: 
    
    LIMIT = 0.1
    
    #hidden_layers is an array of len layers_amount and values: units per layer -->  [units of first layer, units of second layer] 
    def __init__(self, training_set, expected_output, learning_rate, hidden_layers, epochs_amount):
        
        self.training_set = np.array(list(map(lambda t: [1]+t, training_set)))   # add e0 = 1
        self.hidden_layers = list(map(lambda units: units+1, hidden_layers))     # add one unit to each layer --> v0
        self.hidden_layers_amount = len(hidden_layers)
        self.expected_output = expected_output
        self.learning_rate = learning_rate
        self.epochs_amount = epochs_amount
        self.weights = { # weights that connect ei -- 1° hidden layer
            0: np.random.rand(hidden_layers[0], len(training_set)) #filas x cols
        }
        # Dictionary { key:layer value:list de activation states }
        
        # Estados de activacion de cada perceptron desde la capa 0 hasta la capa del output --> len = 1 + len(hidden_layers) + 1
        self.activations = {}

        # Estados de activacion de cada perceptron desde la primera capa oculta hasta la capa del output --> len = len(hidden_layers) + 1
        self.excited_state = {}
        for layer_i in range(self.hidden_layers_amount - 1): #initialize hidden layers weights
            self.weights[layer_i+1] = np.random.rand(hidden_layers[layer_i+1], hidden_layers[layer_i])
        
        self.weights[self.hidden_layers_amount] = np.random.rand(1, hidden_layers[-1]) # connect last hidden layer with the output
        
        # Deltas de cada perceptron desde la primera capa oculta hasta la capa del output --> len = len(hidden_layers) + 1
        self.deltas = {}
        # self.weights[0] --> [[1,2,3, 4],[1,2,3, 4]]
        
        
    def train(self, max_iterations):
         
        for epoch in range(self.epochs_amount): 
            error = 0
            while error > self.LIMIT and iteration < max_iterations: 
                
                i_x = np.random.randint(0, len(training_set))                  # agarro un input random 
                self.apply_input_to_layer_zero(training_set[i_x])              # V0_k = Eu_k
                
                self.propagate()                                               # propagate Vi = g(h) --> calculo las activaciones
                
                last_excited_state = self.excited_state[self.hidden_layers_amount][0]                           
                last_activation_state = self.activations[self.hidden_layers_amount+1][0] # calculate deltas for backpropagate
                self.deltas[self.hidden_layers_amount+1] = [deriv_activation_function(last_excited_state)*(self.expected_output[i_x] - last_activation_state)]       # delta = g' * (expected - real ) se calcula el delta de la capa de salida
                
                self.backpropagate()                                           # retropropagar
                
                self.update_weigths()                                          # a cada neurona le actualizo el peso
                
                error = self.calculate_error() 
            
    
    # Cuando ya agarré el input, le asigno cada componente a el estado de activación de cada unidad de la capa cero 
    def apply_input_to_layer_zero(self, input): # input = [0 1 1 1 0]
        self.activations[0] = input.copy()
   
    # Calculo el estado de activación para todas las neuronas de la red para el input que agarré
    def propagate(self): 
        
        self.activations[self.hidden_layers_amount+1] = []
        self.excited_state[self.hidden_layers_amount] = []
        
        for layer in range(self.hidden_layers_amount - 1): #V[m][i] = activations[layer][unit]
            self.activations[layer+1] = []
            self.excited_state[layer] = []

            # Agregar el nodo adicional para cada capa con estado de activacion = 1
            self.excited_state[layer].append(np.inner(self.weights[layer][0], self.activations[layer]))
            self.activations[layer+1].append(1)
            
            for unit in range(1, self.hidden_layers[layer]): 
                excited_state = np.inner(self.weights[layer][unit], self.activations[layer])   # w[layer][unit]* V[m-1][unit]
                self.excited_state[layer].append(excited_state)
                self.activations[layer+1].append(self.activation_function(excited_state))

        # Activation del ouput:
        excited_state = np.inner(self.weights[self.hidden_layers_amount][0], self.activations[self.hidden_layers_amount])   #con esto se calcula el delta      
    
        self.excited_state[self.hidden_layers_amount].append(excited_state)
        self.activations[self.hidden_layers_amount+1].append(self.activation_function(excited_state))

                         
    def activation_function(self, excited_state):
        return np.tanh(excited_state)

    def deriv_activation_function(self, h):
        return (1 - math.tanh(h)**2)

    # Hago el camino inverso para calcular los deltas
    def backpropagate(self): 
     
        for layer in range(self.hidden_layers_amount+1,1,-1): # delta[layer][unit] voy desde M-1 hasta 2 
            self.deltas[layer] = []
            
            for unit in range(self.hidden_layers[layer]):  
                self.deltas[layer-1].append(deriv_activation_function( self.excited_state[layer-1][unit] )* np.inner(self.weights[layer][unit], self.deltas[layer][unit]))       # delta = g' * (expected - real ) se calcula el delta de la capa de salida
                
    def update_weigths(self): 
        
        for layer in range(self.hidden_layers_amount+1):
            for unit in range(self.hidden_layers[layer]):
                
                delta_w = self.learning_rate *  self.deltas[layer][unit] * self.activations[layer][unit] #deltas arranca de la capa1, activations desde la capa 0 
                self.weights[layer][unit] += delta_w
    
        
    def calculate_error(): 
        error = 0
        for i in range(len(self.training_set)):
            expected = self.expected_output[i]
            excited_state = np.inner(self.weights[self.hidden_layers_amount],self.activations[self.hidden_layers_amount])
            activation_state = activation_function(excited_state)
            error += (expected - activation_state)**2
        return 0.5 * error
        
        
    # capa0     capa1         c2          c3      c4 --> M=4 (no cuenta la cero)   
    # E00  w0   V0=1 w1     V0=1  w2    V0=1  
    # e01      
    # e02       V1=x    
    # e03       V2          V3          V4       O1=Vm
    # e04       V3
    # e05                                        