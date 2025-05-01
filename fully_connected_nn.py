import numpy as np 
import matplotlib.pyplot as plt # Κάνω import τα librarys 
class FullyConnectedNN():
    def __init__(self, input_size=None, hidden_size=None, output_size=None, lr=1e-2):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.lr = lr
         # σε αυτην την μέθοδο αρχικοποιώ τις διαφόρες παραμέτρους του νευρωνικόυ δικτίου, το lr=1e-2 είναι το  το 10-2.


    def init_parameters(self):
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros(self.output_size)
       
#σε αυτήν την σύναρτηση κάνω initialize τις παραμέτρους του νευρωνικού δικτύου και μετα πολλαπλασιασω με το 0.1 απόκλιση σ = 0.1 και κανω χρήση np. random.randn.

    def forward(self, X):
            
            self.z1 = np.dot(X, self.W1) + self.b1
            self.a1 = self.relu(self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.sigmoid(self.z2)
            return self.a2
    # σε αυτήν την μέθοδο κάνω forward pass, δηλαδή υπολογίζω την έξοδο του νευρωνικού δικτύου για μια είσοδο X.


    def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
    

    def relu(self, x):
            return np.maximum(0, x)

#αυτην την μέθοδο δεν την αναφέρει στην εργασία αλλά ειναι η συναρτηση sigmoid που χρησιμοποιίω για την εξοδο του νευρωνικού δικτύου.


    def predict(self, X):
        output_forward_pass = self.forward(X)
        applyforward = np.where(output_forward_pass >=0.5, 1, 0)
        return applyforward

# σε αυτην την συναρτηση κάνω την πρόβλεψη για μια είσοδο X. Χρησιμοποιώ την συνάρτηση np.where για να μετατρέψω τις πιθανότητες σε κλάσεις 0 ή 1 με δυαδική κατηγο-
# ποίηση το 0.5.

    def loss(self,x,y):
        y_pred = self.forward(x)
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss_per_sample = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        total_loss = np.mean(loss_per_sample)
        return total_loss
# σε αυτην την μέθοδο υπολογίζω την απώλεια (loss) του νευρωνικού δικτύου για μια είσοδο x και την αντίστοιχη έξοδο y. 
# Χρησιμοποιώ την συνάρτηση np.clip  υπολογίζω την μέση απώλεια για όλα τα δείγματα.
# το epsilon ειναι μια μικρη τιμή  για να μην έχω ΝΑΝ valueError στην log.


    def backward(self, X, y):
        N = X.shape[0]
        y = y.reshape(self.a2.shape)  # μετατρέπω το y στην μορφή της εξόδου του νευρωνικού δικτύου
        delta2 = self.a2 - y
        dW2 = self.a1.T.dot(delta2) / N
        db2 = np.sum(delta2, axis=0) / N
        delta1 = delta2.dot(self.W2.T) * (self.a1 * (1 - self.a1))
        dW1 = X.T.dot(delta1) / N
        db1 = np.sum(delta1, axis=0,) / N
        self.gradients = \
        {'dW1': dW1, 'db1': db1, 
         'dW2': dW2, 'db2': db2}
         # αποθηκευω τα gradients σε ενα dictionery
        return self.gradients   
    
    # σε αυτην την μέθοδο κάνω backpropagation για να υπολογίσω τα gradients των παραμέτρων του νευρωνικού δικτύου

    #
            
        


    def step(self):
        np.subtract(self.W1, self.lr * self.gradients['dW1'], out=self.W1)
        np.subtract(self.b1, self.lr * self.gradients['db1'], out=self.b1)
        np.subtract(self.W2, self.lr * self.gradients['dW2'], out=self.W2)
        np.subtract(self.b2, self.lr * self.gradients['db2'], out=self.b2)
# σε αυτην την μέθοδο κάνω update τις παραμέτρους του νευρωνικού δικτύου με βάση τα gradients που υπολόγισα στην μέθοδο backward.
# Χρησιμοποιώ την συνάρτηση np.subtract για να κάνω το update, η συναρτήση κάνει αφαίρεση και τα αποθηκευει στην ίδια μεταβλήτη 




    def fit( self , X, y, iterations=10000, batch_size=None, show_step=1000): 
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray) or X.shape[0] != y.shape[0]: # κανω τους ελέχνους
                raise ValueError("X and y must be numpy arrays or the same length ! ")
        self.init_parameters()
        indices = np.random.permutation(X.shape[0]) # κανει shuffle τα indices των δειγμάτων.
        X_shuf = X[indices]
        y_shuf = y[indices]

        N = X.shape[0]  
        for i in range(iterations):
            if batch_size is None:
                X_batch = X
                y_batch = y
            else:
                start = (i * batch_size) % N
                end = start + batch_size
                if end <= N:
                    X_batch = X_shuf[start:end]
                    y_batch = y_shuf[start:end] # εδώ εχω το batch_size
                else:  
                    end_index = np.mod(end, N)
                    X_batch = np.concatenate((X_shuf[start:], X_shuf[:end_index])) # η concatenate ενώνει τα δυο arrays 
                    y_batch = np.concatenate((y_shuf[start:], y_shuf[:end_index]))


            self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.step()


            if (i + 1) % show_step == 0:
                current_loss = self.loss(X_batch, y_batch)
                print(f"Iteration {i+1}, Loss: {current_loss:.4f}")
                
            
# σε αυτην την μέθοδο πραγματοποιω την εκπαίδευση του νευρωνικού δικτύου.
# κανους τους ελέγχους για να δω αν οι είσοδοι ειναι numpy arrays και αν έχουν το ίδιο μήκος
#κάνω τις δίαφορες αρχικοποιήσεις  και μέτα κανω την if αναλογα με το αν εχω batch_size ή οχι.
#μέτα κανω τα  forward, backward και step με την σειρά.
# στην συνέχεια εκτυπώνω την απώλεια.

        #https://medium.com/@shagunmistry/neural-networks-a-simple-introduction-985e4d1cf15f με βοηθήσε το link αυτό