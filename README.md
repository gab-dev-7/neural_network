# Neural Network in C

A simple, modular neural network implementation in pure C. Learn how neural networks work from the ground up with this educational project.

## Quick Start

```bash
# Build the project
make

# Train on XOR problem
make run-xor

# Train on sine wave data
make run

# Train on circle classification
make run-circle
```

## What It Does

This project implements a neural network that can:

- Learn the XOR logic gate
- Approximate mathematical functions (like sine waves)
- Classify points inside/outside a circle
- Use different activation functions (sigmoid, ReLU, tanh, etc.)
- Use different loss functions (MSE, binary cross-entropy)
- Train with momentum and learning rate decay

## Basic Usage

```bash
# Simple XOR training
./nn -d xor -v

# Train on custom data
./nn -d circle -h 16 -b 64 -v

# Show all options
./nn -?
```

## Key Features

### Networks

- 2-layer neural network (input → hidden → output)
- Configurable hidden layer size
- Multiple activation functions
- Batch training with momentum

### Training

- Gradient descent with backpropagation
- Early stopping to prevent overfitting
- Learning rate decay over time
- Weight decay (L2 regularization)
- Progress visualization

### Datasets

- **XOR**: Classic logic gate problem
- **Sine**: sin(x) × cos(y) function
- **Circle**: Inside/outside circle classification
- **Enhanced Circle**: More boundary examples

### Diagnostics

- Gradient checking to verify math is correct
- Train/test split for evaluating performance
- Accuracy calculation for classification

## Example Output

```
training configuration:
  epochs: 5000
  learning rate: 0.100000
  hidden size: 8
  batch size: 32
  dataset: xor
  hidden activation: relu
  output activation: sigmoid
  loss function: mse

training...
epoch 2500/5000 | train loss: 0.043 | test loss: 0.041
final results:
  training accuracy: 98.75%
  test accuracy: 97.50%
```

## Common Examples

```bash
# Quick test with gradient checking
./nn -d xor -g -v

# Larger network for harder problems
./nn -d circle_enhanced -h 32 -b 128 -v

# Try different activations
./nn -d sine -ha tanh -oa linear -v
```

## How It Works

1. **Forward pass**: Input → hidden layer → output layer
2. **Calculate loss**: Compare output to expected values
3. **Backward pass**: Compute gradients using chain rule
4. **Update weights**: Adjust weights using gradients
5. **Repeat**: Process all training data for multiple epochs

## Project Structure

- `neural_network.c/h` - Core network implementation
- `dataset.c/h` - Data generation and handling
- `main.c` - Command line interface and training loop
- `Makefile` - Build automation

---

_Built for education - simple enough to understand, complete enough to be useful._
