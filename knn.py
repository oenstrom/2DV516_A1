import numpy as np

class KNN:
  def set_training(self, x_train, y_train):
    """Set the training data to use."""
    self.X_train = x_train
    self.y_train = y_train
  
  def euc_dist(self, x1, x2, axis=1):
    """Calculate the euclidean distance."""
    return np.linalg.norm(x1 - x2, axis=axis)
  
  def classify(self, k, X_test, axis=1):
    """Classify the X_test values based on the k closest classes."""
    temp_certainty = []
    result = []
    for x_te in X_test:
      distances = self.euc_dist(self.X_train, x_te, axis)
      shortest_idx = distances.argsort()[:k]
      values, counts = np.unique(self.y_train[shortest_idx], return_counts=True, axis=0)
      x_te_class = values[np.argmax(counts)]
      result.append(x_te_class)
      temp_certainty.append(max(counts) / k)
    self.certainty = np.array(temp_certainty)
    return np.array(result)
  
  def regressor(self, k: int, test):
    """Calculate the mean for the k closest y values."""
    result = []
    for t in test:
      closest = self.euc_dist(np.array([self.X_train]).transpose(), np.array([t]))
      result.append(np.mean(self.y_train[closest.argsort()][:k], dtype=np.float64))
    return np.array(result)

  def mse(self, actual_y: np.ndarray, estimated_y: np.ndarray, decimals=-1):
    """Calculate the Mean Squared Error (MSE). Round the answer if decimals is not a negative number."""
    mse = np.divide(np.sum(np.power(actual_y - estimated_y, 2)), np.size(actual_y))
    return mse if decimals < 0 else round(mse, decimals)