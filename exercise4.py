import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

MC_CLASS      = {0.0: "Fail", 1.0: "OK"}
TRAIN         = np.loadtxt("A1_datasets/microchips.csv", delimiter=",", dtype=None)
FAIL          = TRAIN[TRAIN[:, 2] == 0]
OK            = TRAIN[TRAIN[:, 2] == 1]
U_CHIPS       = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])
KNN           = KNeighborsClassifier()

def color_map(x):
  """Map class to colors."""
  return np.array(list(map(lambda c : "green" if c > 0 else "red", x)))

def plot_decision_boundary(ax):
  """Draw the decision boundary on ax and return it."""
  grid_size     = 100
  min_x         = TRAIN[:, 0].min() - 0.1
  max_x         = TRAIN[:, 0].max() + 0.1
  min_y         = TRAIN[:, 1].min() - 0.1
  max_y         = TRAIN[:, 1].max() + 0.1
  xx, yy        = np.meshgrid(np.linspace(min_x, max_x, grid_size), np.linspace(min_y, max_y, grid_size))
  grid_as_test  = np.stack([xx.ravel(), yy.ravel()], axis=1)

  grid_k_nn = KNN.predict(grid_as_test)
  ax.imshow(grid_k_nn.reshape(grid_size, grid_size), origin="lower", extent=(min_x, max_x, min_y, max_y))
  ax.scatter(TRAIN[:, 0], TRAIN[:, 1], marker=".", c=[[0,0,0,0]], edgecolors=color_map(TRAIN[:, 2]))
  return ax

def calculate_training_errors():
  """Calculate the number of training errors."""
  errors = KNN.predict(TRAIN[:, 0:2]) == TRAIN[:, -1]
  return np.size(errors) - np.count_nonzero(errors)

def main():
  """Main function to run when the script is run."""
  KNN.fit(TRAIN[:, 0:2], TRAIN[:, -1])

  plt.figure("Original chip data", figsize=(8, 7))
  plt.suptitle("Original chip data")
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.scatter(FAIL[:, 0], FAIL[:, 1], marker=".", c="red", label="Fail")
  plt.scatter(OK[:, 0], OK[:, 1], marker=".", c="green", label="OK")
  plt.legend()
  
  fig = plt.figure("Decision boundaries and training errors", figsize=(10, 8))
  for i, k in enumerate([1, 3, 5, 7]):
    KNN.n_neighbors = k
    print(f"k = {k}")
    for j, y in enumerate(KNN.predict(U_CHIPS)):
      print(f"  chip{j+1}: [{U_CHIPS[j][0]}, {U_CHIPS[j][1]}] ==> {MC_CLASS.get(y)}")
    
    ax = plot_decision_boundary(fig.add_subplot(221 + i))
    ax.title.set_text(f"k = {k}, training errors = {calculate_training_errors()}")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

  fig.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()