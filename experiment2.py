from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import  model_selection
from fully_connected2_nn import FullyConnectedNN
from sklearn import metrics


import time

start_time = time.time()


data = load_breast_cancer()

X = data.data
y = data.target 
X = np.array(X)
y = np.array(y) # τα κανω numpy arrays .

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=14,shuffle=True) #  κανω shufle για  να είναι  τυχαία τα δεδομένα μου.
scaler = MinMaxScaler() # τα κανω normalization με το MinMaxScaler.
X_train_normal = scaler.fit_transform(X_train)
X_test_normal = scaler.transform(X_test)


print("X type:", type(X), "shape:", X.shape)
print("y type:", type(y), "shape:", y.shape) #βρέσκω τα shape των X και y για να ξέρω τα δεδομένα μου.

n_samples  = X_train_normal.shape[0]   
n_features = X_train_normal.shape[1]   
print(n_samples, n_features)

model = FullyConnectedNN(
    input_size  = n_features,  
    hidden_size = 4,
    output_size = 1,
    lr          = 1e-2 # βάζω αυτο το learning rate γιατί με το 1ε-2 δεν μου δούλευε το μοντέλο.
)

model.fit(
    X_train_normal,
    y_train,
    iterations = 20000,
    batch_size = 64,
    show_step  = 1000
)# κανω fit το μοντέλο μου με τα κανονικοποιημένα δεδομένα και το batch_size 64.
# το iterations το βάζω 20000 για να δω αν θα δουλεψει καλύτερα το μοντέλο μου.
 

 
y_pred = model.predict(X_test_normal)
test_acc = (y_pred.flatten() == y_test).mean()
print(f"Total test accuracy: {test_acc:.2%}")

print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

#  to accurary που έχω ειναι στο 97,08% και ο χρόνος εκπαίδευσης ειναι 1.0265007019042969  sec. εχω μνήμη 32 gb και ryzen 5 5600x 6core 12 threads.