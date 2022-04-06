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
    result = []
    for x_te in X_test:
      distances = self.euc_dist(self.X_train, x_te, axis)
      shortest_idx = distances.argsort()[:k]
      values, counts = np.unique(self.y_train[shortest_idx], return_counts=True, axis=0)
      # print(values)
      # print(counts)
      x_te_class = values[np.argmax(counts)]
      result.append(x_te_class)
    return np.array(result)
  
  def closest_x(self, x, train):
    """Get the closest x values. TODO: Improve, use euc_dist..."""
    temp = np.copy(self.X_train)
    temp[:, 0] = np.abs(temp[:, 0] - x)
    return temp[temp[:, 0].argsort()]

  def regressor(self, k: int, test):
    """Calculate the mean for the k closest y values."""
    result = np.empty((0, 2))
    for t in test:
      result = np.append(result, [[t, np.mean(self.closest_x(t, self.X_train)[:k, -1], axis=0, dtype=np.float64)]], axis=0)
    return result

  def mse(self, actual_y: np.ndarray, estimated_y: np.ndarray, decimals=-1):
    """Calculate the Mean Squared Error (MSE). Round the answer if decimals is not a negative number."""
    mse = np.divide(np.sum(np.power(actual_y - estimated_y, 2)), np.size(actual_y))
    return mse if decimals < 0 else round(mse, decimals)