import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score

# Step 1:  Randomly generates Ntrain training points (i.e. X1, X2 and the corresponding class). Start
# 1 2 3 4 5 6 7
# with Ntrain = 100 and then increase it.
np.random.seed(42)  # Setting seed for reproducibility
Ntrain = 100
X1_train = np.random.uniform(-1, 1, Ntrain)
X2_train = np.random.uniform(-1, 1, Ntrain)
C_train = np.sign(-2 * np.sign(X1_train) * np.abs(X1_train)**(2/3) + 4 * X2_train)

# 2. Plots X2 versus X1 with a red point if the class is -1, with a blue point if the class is +1.
plt.scatter(X1_train[C_train == -1], X2_train[C_train == -1], color='red', label='Class -1')
plt.scatter(X1_train[C_train == 1], X2_train[C_train == 1], color='blue', label='Class +1')
plt.title('Training Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# 3. Generates the C4.5 decision tree using these training points.
X_train = np.column_stack((X1_train, X2_train))
clfX = tree.DecisionTreeClassifier(criterion='entropy')
clfX = clfX.fit(X_train, C_train)

# 4. Tests the decision tree on other Ntest = 20000 randomly taken samples, generating for each of them the estimated class Cˆ.
Ntest = 20000
X1_test = np.random.uniform(-1, 1, Ntest)
X2_test = np.random.uniform(-1, 1, Ntest)
X_test = np.column_stack((X1_test, X2_test))
C_test = np.sign(-2 * np.sign(X1_test) * np.abs(X1_test)**(2/3) + 4 * X2_test)


# Step 5: 5. For the test dataset, plots X2 versus X1 with a red point if Cˆ = −1, with a blue point if Cˆ = 1.
hatC_test = clfX.predict(X_test)

plt.scatter(X1_test[hatC_test == -1], X2_test[hatC_test == -1], color='red', label='Predicted Class -1')
plt.scatter(X1_test[hatC_test == 1], X2_test[hatC_test == 1], color='blue', label='Predicted Class +1')
plt.title('Test Predictions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# Step 7: Measure accuracy on the test dataset
accuracy = accuracy_score(C_test, hatC_test)
print('Accuracy =', accuracy)