import numpy as np
import matplotlib.pyplot as plt
import h5py

def linear_value(iteration, max_iter, min_val, max_val):
    """
    Calculate the linearly decreasing value from max_val to min_val.
    
    Arguments:
    iteration -- Current iteration count (starting at 0)
    max_iter -- Maximum number of iterations
    min_val -- Min value
    max_val -- Max value
    
    Returns:
    value -- Value at the current iteration
    """
    # Compute the normalized position in the cycle.
    normalized_position = iteration / max_iter  # Scale to [0, 1]
    
    # Compute the linear value
    value = max_val - (max_val - min_val) * normalized_position
    
    return value

def plot_mutation_parameters(max_iter=20000):
    iterations = np.arange(max_iter)
    mutation_scale = [0.008] * max_iter
    mutation_rate = [linear_value(i, max_iter, 0.1, 0.9) for i in iterations]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Plot MUTATION RATE (LEFT Y-AXIS) ---
    color_rate = 'tab:red'
    ax1.plot(iterations, mutation_rate, 
             color=color_rate, 
             linestyle='--', 
             marker='s', 
             markersize=4,
             markevery=1000,
             label='Mutation Rate (Dynamic)')
    
    # Configure left axis
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Mutation Rate', color=color_rate)
    ax1.tick_params(axis='y', labelcolor=color_rate)
    ax1.set_ylim(0, 1.0)  # Start at 0
    ax1.set_xlim(0, max_iter)  # X-axis from 0 to 20,000

    # --- Plot MUTATION SCALE (RIGHT Y-AXIS) ---
    ax2 = ax1.twinx()
    color_scale = 'tab:blue'
    ax2.plot(iterations, mutation_scale, 
             color=color_scale, 
             linestyle='-', 
             marker='o', 
             markersize=4,
             markevery=1000,
             label='Mutation Scale (Constant)')
    
    # Configure right axis
    ax2.set_ylabel('Mutation Scale', color=color_scale)
    ax2.tick_params(axis='y', labelcolor=color_scale)
    ax2.set_ylim(0, 0.01)  # Start at 0

    # --- Align axes at (0,0) and customize grid ---
    # Move spines to intersect at (0,0)
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    
    # Hide top/right spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)  # Hide right spine for cleaner look

    # Add square grid
    ax1.grid(True, linestyle='--', alpha=0.3, which='both', axis='both')

    # Title and legend
    plt.title('Mutation Parameters Over 20,000 Iterations')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
    plt.show()

    # The plot_costs plots a visualization comparing the evolution of the best costs for the leader and follower solutions over time.
# ==================== BASIC FUNCTIONS ====================

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    """
    Z = np.dot(W, A) + b
    return Z

def relu(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)

def sigmoid(Z):
    """Sigmoid activation function with numerical stability."""
    Z = np.clip(Z, -500, 500)  # Limit Z to prevent exp(-Z) from overflowing
    return 1 / (1 + np.exp(-Z))


# ==================== FORWARD PROPAGATION WITH DROPOUT ====================

def linear_activation_forward(A_prev, W, b, activation="relu", keep_prob=1.0, training=True):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer with Dropout.

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    keep_prob -- probability of keeping a neuron active during training (drop-out probability)
    training -- boolean, if True, apply dropout; if False, do not apply dropout

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    """
    # 1. Compute linear and activation
    if activation == "sigmoid":
        Z = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
    elif activation == "relu":
        Z = linear_forward(A_prev, W, b)
        A = relu(Z)
    else:
        raise ValueError("Unsupported activation function!")
        
    # 2. Apply Dropout 
    if training and keep_prob < 1.0:
        D = np.random.rand(A.shape[0], A.shape[1])     # Create random matrix D
        D = (D < keep_prob).astype(float)              # Convert to 0 or 1 (Mask)
        A = A * D                                      # Shut down some neurons randomly
        A = A / keep_prob                              # Inverted Dropout: scale up to maintain expected value
        
    return A


def L_model_forward(X, parameters, keep_prob=1.0, training=True):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
    Supports Dropout for hidden layers.

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    keep_prob -- probability of keeping a neuron active (drop-out probability)
    training -- boolean, True for training phase, False for testing/inference phase

    Returns:
    AL -- activation value from the output (last) layer
    """
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    # 1. [LINEAR -> RELU] * (L-1) with Dropout
    for l in range(1, L):
        A_prev = A 
        A = linear_activation_forward(
            A_prev, 
            parameters["W" + str(l)], parameters["b" + str(l)], 
            activation="relu", 
            keep_prob=keep_prob, 
            training=training
        )
        
    # 2. LINEAR -> SIGMOID (Output layer does NOT apply Dropout)
    AL = linear_activation_forward(
        A, 
        parameters["W" + str(L)], parameters["b" + str(L)], 
        activation="sigmoid", 
        keep_prob=1.0,  # Always 1.0 for output layer
        training=training
    )
    
    # Clip AL to prevent log(0)
    AL = np.clip(AL, 1e-10, 1 - 1e-10)
    assert(AL.shape == (1, X.shape[1]))
  
    return AL


# ==================== COST COMPUTATION WITH L2 REGULARIZATION ====================

def compute_cost(AL, Y, parameters=None, lambd=0.0):
    """
    Implement the cost function with Cross-Entropy and optional L2 Regularization.
    
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector, shape (1, number of examples)
    parameters -- dictionary containing model parameters (W1, b1, ..., WL, bL)
    lambd -- regularization hyperparameter (lambda), 0.0 means no L2 regularization
    
    Returns:
    cost -- cross-entropy cost + L2 regularization cost
    """
    m = Y.shape[1]
    
    # 1. Original Cross-entropy cost
    cost = (-1/m) * np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), (1-Y)))
    
    # 2. L2 Regularization cost
    if lambd > 0 and parameters is not None:
        L = len(parameters) // 2
        sum_W_squared = 0
        for l in range(1, L + 1):
            sum_W_squared += np.sum(np.square(parameters["W" + str(l)]))
        
        L2_regularization_cost = (lambd / (2 * m)) * sum_W_squared
        cost = cost + L2_regularization_cost
        
    cost = np.squeeze(cost)      # Make sure cost is a scalar
    
    return cost
def predict_results(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas = L_model_forward(X, parameters, )

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


def plot_costs(cost_1, cost_2, max_layer_dims, lambd=0.0, keep_prob=1.0):
    """
    Plots a visualization comparing the evolution of the best costs 
    for the leader and follower solutions over time, including hyperparameters.
    """
    # Plot with different colors AND markers
    plt.plot(np.squeeze(cost_1), 
             color='red', 
             marker='o',         # Circle marker
             markersize=4,       # Marker size
             markevery=10,       # Show marker every 10th point
             label='Leader Cost')
    
    plt.plot(np.squeeze(cost_2), 
             color='blue', 
             marker='s',         # Square marker
             markersize=4, 
             markevery=10, 
             label='Follower Cost')

    # Customize plot
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    
    plt.title(f"Dual-Individual GA: Leader vs. Follower\n"
              f"Max_layer_dims = {max_layer_dims}\n"
              f"λ (L2) = {lambd:.2e} | Keep Prob = {keep_prob:.2f}")
    
    plt.legend(loc='best')       # Auto-placement for legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()



def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
