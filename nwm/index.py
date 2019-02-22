import neurons_on_caffeine as noc
import numpy as np

model = noc.Model(input_nodes=2)

model.add_layer(4)
model.add_layer(1)

X = np.array([[1, 0]])

model.train(X=X, y=0, iterations=1000, alpha=0.1)

print(int(model.predict(X)))
