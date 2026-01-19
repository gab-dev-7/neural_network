#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// activation function types
typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_TANH,
    ACTIVATION_LINEAR,
    ACTIVATION_LEAKY_RELU,
    ACTIVATION_SOFTMAX
} ActivationType;

// loss function types
typedef enum {
    LOSS_MSE,
    LOSS_BINARY_CE, // binary cross-entropy
    LOSS_MAE        // mean absolute error
} LossType;

// neural network structure
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;

    // activation types
    ActivationType hidden_activation;
    ActivationType output_activation;
    LossType loss_type;

    // weights and biases
    double** w1;
    double* b1;
    double** w2;
    double* b2;

    // activations
    double* hidden;
    double* output;
    double* z_hidden; // pre-activation for hidden layer
    double* z_output; // pre-activation for output layer

    // gradients (accumulated over batch)
    double** dw1;
    double* db1;
    double** dw2;
    double* db2;

    // momentum
    double** prev_dw1;
    double** prev_dw2;
    double momentum;

    // regularization
    double weight_decay;
} NeuralNetwork;

// activation functions
double sigmoid(double x);
double relu(double x);
double leaky_relu(double x);
double tanh_activation(double x);
double linear(double x);

// activation derivatives
double sigmoid_derivative_from_output(double output);
double relu_derivative(double x);
double leaky_relu_derivative(double x);
double tanh_derivative(double x);
double linear_derivative(void);

// get activation function and derivative based on type
double activate(double x, ActivationType type);
double activate_derivative(double x, ActivationType type);

// loss functions
double mse_loss(double output, double target);
double binary_cross_entropy_loss(double output, double target);
double mae_loss(double output, double target);
double compute_loss(double output, double target, LossType type);
double compute_loss_derivative(double output, double target, LossType type);

NeuralNetwork* create_network(int input_size, int hidden_size, int output_size,
                              ActivationType hidden_activation, ActivationType output_activation,
                              LossType loss_type);
void forward(NeuralNetwork* nn, double* input);

// batch processing functions
void backward_accumulate(NeuralNetwork* nn, double* input, double* target);
void update_weights(NeuralNetwork* nn, double learning_rate, int batch_size);
void reset_gradients(NeuralNetwork* nn);

double calculate_loss(NeuralNetwork* nn, double** inputs, double** targets, int num_samples);
void free_network(NeuralNetwork* nn);
double gradient_check(NeuralNetwork* nn, double* input, double* target, double epsilon);

// training function
double train_epoch(NeuralNetwork* nn, double** inputs, double** targets,
                   int num_samples, int batch_size, double learning_rate, int shuffle);

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
    int shuffle;

    // learning rate scheduling
    double decay_rate;
    int decay_steps;

    // dataset parameters
    char dataset_type[20];
    int dataset_size;
    double train_test_split;
    double boundary_ratio; // for enhanced circle dataset

    // activation function parameters
    char hidden_activation[20];
    char output_activation[20];

    // loss function
    char loss_function[20];

    // regularization
    double weight_decay;

    // advanced features
    int use_validation_set;
    double validation_ratio;
    int use_enhanced_circle;
} TrainingConfig;

// cli and utility functions
void print_usage(const char* program_name);
TrainingConfig parse_arguments(int argc, char* argv[]);
void print_config(const TrainingConfig* config);
void print_progress(int epoch, int total_epochs, double error);

#endif
