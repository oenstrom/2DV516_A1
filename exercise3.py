from mnist import mnist
import numpy as np
import knn

def main():
    """Main function to run when the script is run."""
    # plt.imshow(train_images[0].reshape(28, 28), cmap=plt.get_cmap("gray"))

    train_images, train_labels, test_images, test_labels = mnist("A1_datasets")
  
    # train_images = train_images[:10000]
    # train_labels = train_labels[:10000]
    # test_images = test_images[:1000]
    # test_labels = test_labels[:1000]

    KNN = knn.KNN()
    KNN.set_training(train_images, train_labels)

    for k in range(1, 16, 2):
        result = KNN.classify(k, test_images)
        print(f"k = {k}")
        print("  Accuracy:", round(np.count_nonzero((result == test_labels).all(axis=1)) / np.size(test_labels, axis=0), 3))
        print("  Average certainty:", round(np.mean(KNN.certainty), 3))
        print()

if __name__ == "__main__":
    main()