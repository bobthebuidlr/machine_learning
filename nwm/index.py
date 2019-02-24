import neurons_on_caffeine as noc
import numpy as np

model = noc.Model(input_nodes=2)

model.add_layer(3)
model.add_layer(1)

X = np.array([[[0], [0]], [[1], [1]], [[0], [1]], [[1], [0]]])
y = np.array([[0], [0], [1], [1]])

model.train(X=X, y=y, alpha=0.1, iterations=100000)

print(model.predict(X[0]))
print(model.predict(X[1]))
print(model.predict(X[2]))
print(model.predict(X[3]))
