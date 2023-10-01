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
python test.py --epochs <number_of_epochs> --batch-size <batch_size> --save-model
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
