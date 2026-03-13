import numpy as np

# --- 1. Data Generation (Specific Problem: Hiring Prediction) ---
def generate_hiring_data(samples=100):
    np.random.seed(0)
    # Generate random scores for 2 features (Aptitude, Interview)
    X = np.random.rand(samples, 2) 
    y = np.zeros((samples, 1))
    
    # Logic: Hired if (Aptitude + Interview > 1.0) AND (Interview > 0.4)
    # This creates a non-linear decision boundary
    for i in range(samples):
        score_sum = X[i, 0] + X[i, 1]
        if score_sum > 1.0 and X[i, 1] > 0.4:
            y[i, 0] = 1 # Hired
        else:
            y[i, 0] = 0 # Rejected
            
    return X, y

# --- 2. Activation Functions ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return z > 0

# --- 3. Neural Network Class ---
class HiringPredictorNN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        np.random.seed(42)
        # Layer 1: Input -> Hidden 1
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.b1 = np.zeros((1, hidden1_size))
        
        # Layer 2: Hidden 1 -> Hidden 2
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.b2 = np.zeros((1, hidden2_size))
        
        # Layer 3: Hidden 2 -> Output
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1) # ReLU activation
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2) # ReLU activation
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3) # Sigmoid for binary output
        return self.A3

    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        
        # Calculate Gradients (Backpropagation)
        # Layer 3 (Output)
        dZ3 = self.A3 - y
        dW3 = (1/m) * np.dot(self.A2.T, dZ3)
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
        
        # Layer 2 (Hidden)
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_deriv(self.Z2)
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Layer 1 (Hidden)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_deriv(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update Parameters
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def train(self, X, y, epochs=20000, learning_rate=0.1):
        for i in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            
            if i % 2000 == 0:
                loss = -np.mean(y * np.log(self.A3) + (1 - y) * np.log(1 - self.A3))
                print(f"Epoch {i}, Loss: {loss:.4f}")

# --- 4. Driver Code ---
if __name__ == "__main__":
    # Generate 200 candidate profiles
    X, y = generate_hiring_data(samples=200)
    
    print(f"Dataset Shape: {X.shape}")
    print("Training Hiring Predictor Model...")

    # Initialize: 2 Inputs -> 5 Hidden -> 4 Hidden -> 1 Output
    model = HiringPredictorNN(input_size=2, hidden1_size=5, hidden2_size=4, output_size=1)
    
    # Train
    model.train(X, y, epochs=20000, learning_rate=0.05)

    # Test on a specific new candidate
    # Candidate A: Aptitude 0.9, Interview 0.8 (Should be Hired)
    # Candidate B: Aptitude 0.3, Interview 0.2 (Should be Rejected)
    test_candidates = np.array([[0.9, 0.8], [0.3, 0.2]])
    predictions = model.forward(test_candidates)
    
    print("\nTest Results:")
    print(f"Candidate A (High Scores): Prediction {predictions[0][0]:.4f} -> {'Hired' if predictions[0]>0.5 else 'Rejected'}")
    print(f"Candidate B (Low Scores):  Prediction {predictions[1][0]:.4f} -> {'Hired' if predictions[1]>0.5 else 'Rejected'}")