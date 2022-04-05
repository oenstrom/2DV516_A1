import matplotlib.pyplot as plt
import numpy as np
import knn

MC_CLASS      = {0.0: "Fail", 1.0: "OK"}
TRAIN         = np.loadtxt("A1_datasets/microchips.csv", delimiter=",", dtype=None)
FAIL          = TRAIN[TRAIN[:, 2] == 0]
OK            = TRAIN[TRAIN[:, 2] == 1]
U_CHIPS       = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])
GRID_SIZE     = 100
MIN_X         = TRAIN[:, 0].min() - 0.1
MAX_X         = TRAIN[:, 0].max() + 0.1
MIN_Y         = TRAIN[:, 1].min() - 0.1
MAX_Y         = TRAIN[:, 1].max() + 0.1
A             = np.linspace(MIN_X, MAX_X, GRID_SIZE)
B             = np.linspace(MIN_Y, MAX_Y, GRID_SIZE)
XX, YY        = np.meshgrid(A, B)
GRID_AS_TEST  = np.stack([XX.ravel(), YY.ravel()], axis=1)
KNN = knn.KNN()

def color_map(x):
  """Map class to colors."""
  return np.array(list(map(lambda c : "green" if c > 0 else "red", x)))

def plot_decision_boundary(ax, k):
  """Draw the decision boundary on ax and return it."""
  grid_k_nn = KNN.classify(k, GRID_AS_TEST)
  ax.imshow(grid_k_nn.reshape(GRID_SIZE, GRID_SIZE), origin="lower", extent=(MIN_X, MAX_X, MIN_Y, MAX_Y))
  ax.scatter(TRAIN[:, 0], TRAIN[:, 1], marker=".", c=[[0,0,0,0]], edgecolors=color_map(TRAIN[:, 2]))
  return ax

def calculate_training_errors(k, train_as_test):
  """Calculate the number of training errors."""
  errors = train_as_test[:, -1] == KNN.classify(k, train_as_test[:, 0:2])
  return np.size(errors) - np.count_nonzero(errors)

def main():
  """Main function to run when the script is run."""
  KNN.set_training(TRAIN[:, 0:2], TRAIN[:, -1])

  plt.figure("Original chip data", figsize=(8, 7))
  plt.suptitle("Original chip data")
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.scatter(FAIL[:, 0], FAIL[:, 1], marker=".", c="red", label="Fail")
  plt.scatter(OK[:, 0], OK[:, 1], marker=".", c="green", label="OK")
  plt.legend()

  fig = plt.figure("Decision boundaries and training errors", figsize=(10, 8))
  for i, k in enumerate([1, 3, 5, 7]):
    print("k =", k)
    for j, chip_y in enumerate(KNN.classify(k, U_CHIPS)):
      print(f"  chip{j+1}: [{U_CHIPS[j][0]}, {U_CHIPS[j][1]}] ==> {MC_CLASS.get(chip_y)}")

    ax = plot_decision_boundary(fig.add_subplot(221 + i), k)
    ax.title.set_text(f"k = {k}, training errors = {calculate_training_errors(k, TRAIN)}")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

  fig.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()