from mnist import mnist
import numpy as np
import matplotlib.pyplot as plt
import knn

def main():
  """Main function to run when the script is run."""
  # plt.imshow(train_images[0].reshape(28, 28), cmap=plt.get_cmap("gray"))

  train_images, train_labels, test_images, test_labels = mnist("A1_datasets")
  
  # train_images = train_images[10:19]
  # train_labels = train_labels[10:19]
  
  # train_images = train_images[:10000]
  # train_labels = train_labels[:10000]
  

  # plt.imshow(test_images[0].reshape(28, 28), cmap=plt.get_cmap("gray"))
  # plt.show()

  KNN = knn.KNN()
  KNN.set_training(train_images, train_labels)
  print(KNN.X_train.shape)
  print(KNN.y_train.shape)
  print(train_images)
  print(test_images[0].shape)

  result = KNN.classify(3, test_images)
  print(result)
  exit()
  # for i, img in enumerate(test_images):
  #   label = test_labels[i]


  img = test_images[1337]
  lab = test_labels[1337]
  awd = KNN.classify(500, [img])
  plt.figure()
  plt.suptitle(f"Predicted: {awd} \n Correct: {lab}")
  plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap("gray"))

  img = test_images[238]
  lab = test_labels[238]
  awd = KNN.classify(500, [img])
  plt.figure()
  plt.suptitle(f"Predicted: {awd} \n Correct: {lab}")
  plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap("gray"))

  img = test_images[299]
  lab = test_labels[299]
  awd = KNN.classify(500, [img])
  plt.figure()
  plt.suptitle(f"Predicted: {awd} \n Correct: {lab}")
  plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap("gray"))

  # plt.figure("The matching one from X_train")
  # plt.imshow(KNN.X_train[awd[0]].reshape(28, 28), cmap=plt.get_cmap("gray"))
  # print(awd)
  # print(KNN.y_train[awd[0]])
  # plt.figure()
  # for i in range(9):
  #   plt.subplot(331 + i)
  #   plt.imshow(train_images[i].reshape(28, 28), cmap=plt.get_cmap("gray"))
  # # print(train_labels)
  plt.show()
  # hej = train_images[0].reshape(28, 28)
  # hej = hej.reshape(392, 2)
  # plt.imshow(hej.reshape(28, 28), cmap=plt.get_cmap("gray"))
  # print(hej)
  # plt.show()


if __name__ == "__main__":
  main()