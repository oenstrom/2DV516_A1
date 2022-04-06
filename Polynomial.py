import matplotlib.pyplot as plt
import numpy as np
import knn

DATA         = np.loadtxt("A1_datasets/polynomial200.csv", delimiter=",", dtype=None)
TRAINING_SET = DATA[0:100]
TEST_SET     = DATA[100:]

def main():
  """Main function to run when the script is run."""
  fig = plt.figure("Training set and Test set", figsize=(12, 6))
  
  ax1 = fig.add_subplot(121)
  ax1.title.set_text("Training data")
  ax1.set_xlabel("x")
  ax1.set_ylabel("y")
  ax1.scatter(TRAINING_SET[:, 0], TRAINING_SET[:, 1], marker=".")
  
  ax2 = fig.add_subplot(122)
  ax2.title.set_text("Test data")
  ax2.set_xlabel("x")
  ax2.set_ylabel("y")
  ax2.scatter(TEST_SET[:, 0], TEST_SET[:, 1], marker=".")

  KNN = knn.KNN()
  KNN.set_training(TRAINING_SET[:, 0], TRAINING_SET[:, 1])

  fig2     = plt.figure("k-NN regression and training errors", figsize=(12, 6))
  x_values = np.linspace(1, 25, 200)
  for i, k in enumerate([1, 3, 5, 7, 9, 11]):
    ax = fig2.add_subplot(231 + i)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    est_train_y = KNN.regressor(k, TRAINING_SET[:, 0])
    ax.title.set_text(f"k = {k}, MSE = {KNN.mse(TRAINING_SET[:, 1], est_train_y, 2)}")
    
    regression_y = KNN.regressor(k, x_values)
    ax.scatter(TRAINING_SET[:, 0], TRAINING_SET[:, 1], marker=".", c="darkblue")
    ax.plot(x_values, regression_y, color="lightblue")

    est_test_y = KNN.regressor(k, TEST_SET[:, 0])
    print(f"MSE test error for k = {k}:", KNN.mse(TEST_SET[:, 1], est_test_y, 2), sep="\n  ")

  fig2.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()