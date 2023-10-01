# MNIST PoC using TensorFlow

This project contains a Python script that trains a neural network model using the MNIST dataset. The trained model is then used to make predictions, and its performance is evaluated and plotted.

## Features
- Load and preprocess the MNIST dataset
- Build and train a Sequential model using TensorFlow and Keras
- Evaluate the model's accuracy and plot training history
- Make predictions using the trained model
- Enhanced logging for tracking the training process and debugging
- Command-line arguments for customization of epochs, batch size, and saving the model
- Option to save the trained model

## Requirements
- Python
- TensorFlow
- Matplotlib
- NumPy

## Usage
### Running the Script
To run the script, use the following command:
```bash
python poc.py --epochs <number_of_epochs> --batch-size <batch_size> --save-model
```

### Command-line Arguments
- `--epochs` (optional): Specify the number of epochs for training the model (default is 10).
- `--batch-size` (optional): Specify the batch size for training (default is 32).
- `--save-model` (optional): Include this flag to save the trained model.

## Log Files
The script logs events and errors into a log file named `mnist_poc_<timestamp>.log`, located in the same directory as the script.

## Example
```bash
python test.py --epochs 20 --batch-size 64 --save-model
```

This command will run the script for 20 epochs, with a batch size of 64, and will save the trained model after completion.

## Output
The script will output the model training and evaluation process, including the accuracy and loss plots. If any errors occur during the execution, they will be printed to the console and logged to the log file.

```
root@localhost:~/poc# python poc.py --save-model

--- Loading and Preprocessing Data ---
The MNIST dataset is composed of grayscale images of handwritten digits, each image is labeled with the digit it represents.

Data has been successfully loaded and normalized!

--- Creating the Model ---
A sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
Model has been created successfully!

--- Compiling and Training the Model ---
During compilation, we configure the model with the optimizer, loss function, and metrics.
Model has been compiled successfully!
Training the model for 10 epochs with batch size 32
Epoch 1/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2566 - accuracy: 0.9279 - val_loss: 0.1339 - val_accuracy: 0.9594
Epoch 2/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1123 - accuracy: 0.9664 - val_loss: 0.0914 - val_accuracy: 0.9720
Epoch 3/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0771 - accuracy: 0.9766 - val_loss: 0.0840 - val_accuracy: 0.9747
Epoch 4/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0580 - accuracy: 0.9822 - val_loss: 0.0751 - val_accuracy: 0.9768
Epoch 5/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0452 - accuracy: 0.9862 - val_loss: 0.0868 - val_accuracy: 0.9735
Epoch 6/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0351 - accuracy: 0.9888 - val_loss: 0.0742 - val_accuracy: 0.9781
Epoch 7/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0282 - accuracy: 0.9916 - val_loss: 0.0709 - val_accuracy: 0.9784
Epoch 8/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0229 - accuracy: 0.9926 - val_loss: 0.0743 - val_accuracy: 0.9769
Epoch 9/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0184 - accuracy: 0.9941 - val_loss: 0.0773 - val_accuracy: 0.9790
Epoch 10/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0153 - accuracy: 0.9952 - val_loss: 0.0778 - val_accuracy: 0.9777
Model has been trained successfully!

--- Evaluating the Model and Plotting the Training History ---
313/313 - 0s - loss: 0.0778 - accuracy: 0.9777 - 358ms/epoch - 1ms/step

Test accuracy: 0.9776999950408936

--- Making Predictions and Plotting ---
313/313 [==============================] - 0s 1ms/step

--- Saving the Model ---
Model has been saved successfully!
```
