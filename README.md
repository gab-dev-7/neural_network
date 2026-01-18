# Neural Network in C

A simple neural network implementation in C for learning XOR and other binary functions.

## Features

- Feedforward neural network with backpropagation
- Configurable via command-line arguments
- Progress visualization
- Memory-safe with proper cleanup
- Makefile for easy building

## Building

```bash
# Build the program
make

# Build and run
make run

# Clean build files
make clean
```

## Usage

```bash
# Basic usage (default parameters)
./neural_network

# With custom parameters
./neural_network -e 5000 -l 0.3 -h 8 -v

# Show help
./neural_network --help
```

## Command-line Options

- `-e, --epochs NUM`: Number of training epochs (default: 10000)
- `-l, --learning-rate FLOAT`: Learning rate (default: 0.5)
- `-h, --hidden-size NUM`: Hidden layer size (default: 4)
- `-b, --batch-size NUM`: Batch size (default: 1)
- `-v, --verbose`: Show detailed training progress

## Project Structure

- `neural_network.h`: Header file with declarations
- `neural_network.c`: Neural network implementation
- `main.c`: Main program with CLI argument parsing
- `Makefile`: Build automation
