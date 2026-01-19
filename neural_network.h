#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// neural network structure
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;

    // weights and biases
    double** w1;
    double* b1;
    double** w2;
    double* b2;

    // activations
    double* hidden;
    double* output;

    // gradients
    double** dw1;
    double* db1;
    double** dw2;
    double* db2;

    // momentum
    double** prev_dw1;
    double** prev_dw2;
    double momentum;
} NeuralNetwork;

// function declarations
double sigmoid(double x);
double sigmoid_derivative_from_output(double x);

NeuralNetwork* create_network(int input_size, int hidden_size, int output_size);
void forward(NeuralNetwork* nn, double* input);
void backward(NeuralNetwork* nn, double* input, double* target,
              double learning_rate);
double calculate_mse(NeuralNetwork* nn, double** inputs, double** targets,
                     int num_samples);
void free_network(NeuralNetwork* nn);

double gradient_check(NeuralNetwork* nn, double* input, double* target, double epsilon);

// training configuration
typedef struct {
    int epochs;
    double learning_rate;
    int hidden_size;
    int batch_size;
    int verbose;
    double validation_split;
    int patience;
    double momentum;
    int gradient_check;
} TrainingConfig;

// cli and utility functions
void print_usage(const char* program_name);
TrainingConfig parse_arguments(int argc, char* argv[]);
void print_config(const TrainingConfig* config);
void print_progress(int epoch, int total_epochs, double error);

#endif
