import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import logging

# Setup Logging
logging.basicConfig(filename='mnist_poc.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='MNIST PoC using TensorFlow')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training the model')

args = parser.parse_args()
EPOCHS = args.epochs


def load_and_preprocess_data():
    logging.info("Loading and preprocessing data")
    print("\n--- Loading and Preprocessing Data ---")
    # Additional explanations for beginners
    print("The MNIST dataset is composed of grayscale images of handwritten digits, each image is labeled with the digit it represents.")
    try:
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        print("\nData has been successfully loaded and normalized!")
        return (train_images, train_labels), (test_images, test_labels)
    except Exception as e:
        print(f"\nError occurred: {e}")
        logging.error("Error in loading and preprocessing data", exc_info=True)
        raise


def create_model():
    logging.info("Creating the Model")
    print("\n--- Creating the Model ---")
    print("A sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.")
    try:
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)
        ])
        print("Model has been created successfully!")
        return model
    except Exception as e:
        print(f"\nError occurred: {e}")
        logging.error("Error in creating the model", exc_info=True)
        raise


def compile_and_train_model(model, train_images, train_labels, test_images, test_labels, epochs):
    logging.info("Compiling and training the model")
    print("\n--- Compiling and Training the Model ---")
    print("During compilation, we configure the model with the optimizer, loss function, and metrics.")
    try:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        print("Model has been compiled successfully!")
        
        print(f"Training the model for {epochs} epochs")
        history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
        print("Model has been trained successfully!")
        return history
    except Exception as e:
        print(f"\nError occurred: {e}")
        logging.error("Error in compiling and training the model", exc_info=True)
        raise


def evaluate_and_plot(model, test_images, test_labels, history):
    logging.info("Evaluating the model and plotting the training history")
    print("\n--- Evaluating the Model and Plotting the Training History ---")
    try:
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f'\nTest accuracy: {test_acc}')
        
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
    except Exception as e:
        print(f"\nError occurred: {e}")
        logging.error("Error in evaluating and plotting", exc_info=True)
        raise


def make_predictions_and_plot(model, test_images, test_labels):
    logging.info("Making predictions and plotting")
    print("\n--- Making Predictions and Plotting ---")
    try:
        predictions = model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        plt.figure(figsize=(10,5))
        for i in range(15):
            plt.subplot(3,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(test_images[i], cmap=plt.cm.binary)
            plt.xlabel(f"True: {test_labels[i]}\nPred: {predicted_labels[i]}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\nError occurred: {e}")
        logging.error("Error in making predictions and plotting", exc_info=True)
        raise


def main():
    try:
        (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
        model = create_model()
        history = compile_and_train_model(model, train_images, train_labels, test_images, test_labels, EPOCHS)
        evaluate_and_plot(model, test_images, test_labels, history)
        make_predictions_and_plot(model, test_images, test_labels)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.error("Error in main", exc_info=True)


if __name__ == "__main__":
    main()
