from sklearn import datasets, metrics, preprocessing, model_selection
from sklearn.metrics import accuracy_score
import numpy as np
from  generate_datasets import generate_binary_problem
from sklearn.model_selection import train_test_split
from  helpers import plot_decision_boundary
import numpy as np
from fully_connected_nn import FullyConnectedNN
from generate_datasets import generate_flower_problem


centers = np.array([[0, 0], [8, 8]])
X, y = generate_binary_problem(centers, 1000)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=11)
#κανω split και τεστ τα δεδομένα μου σε train και test με το 30% να ειναι το test set και το 70% το train set.
model = FullyConnectedNN(input_size=2, hidden_size=4, output_size=1, lr=1e-2)
model.fit(X_train, y_train, iterations=10000, batch_size=None, show_step=1000) 
#οπως λεει και η εργασια κανω call της 2 συναρτησης και λεει μονο για το batch_size=None, τα αλλα τα δοκιμάζω εγω.
y_pred_test = model.predict(X_test).reshape(-1)
test_acc = np.mean(y_pred_test == y_test)
print(f"Test accuracy: {test_acc*100:.2f}%")


plot_decision_boundary(lambda x: model.predict(x).flatten(), X_train, y_train) # κανω  plot την  αποφαση του μοντελου μου.


# το loss μου βγαινει σταθέρο 0,69 και το accuraray μου βγαίνει 50% 




X, y = generate_flower_problem()

X = np.transpose(X) if X.shape[0] == 2 else X # το κανω 2d

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=11
)

model = FullyConnectedNN(
    input_size=2,
    hidden_size=4,
    output_size=1,
    lr=1e-2 
)
model.fit(
    X_train,
    y_train,
    iterations=10000,
    batch_size=None,
    show_step=1000
)
# δεν βάζω batch_size γιατι δεν το ζητάει η εργασία και το εχω βάλει στο None

y_pred = model.predict(X_test)
test_acc = (y_pred.squeeze() == y_test).mean() # βρισκω το accuracy του test set.
print(f"test accuracy: {test_acc:.2%}") 

plot_decision_boundary(lambda x: model.predict(x).flatten(), X_train, y_train)



#Τί  παρατηρείτε; Πώς θα μπορούσε να λυθεί το συγκεκριμένο πρόβλημα και τί θα αλλάζατε;


#####  το τεστ accuracy είναι περιπού στο 63%, το loss είναι περιπόυ στο 69-72%,θα αύξανα τους νευρόνες ή θα έβαζα μια relu.
