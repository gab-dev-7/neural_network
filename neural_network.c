#include "neural_network.h"

// Activation function (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize neural network
NeuralNetwork* create_network(int input_size, int hidden_size, int output_size) {

    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        perror("Failed to allocate neural network");
        exit(EXIT_FAILURE);
    }

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    // Seed random number generator
    srand(time(NULL));

    // Allocate memory for weights and biases
    nn->w1 = (double**)malloc(hidden_size * sizeof(double*));
    nn->b1 = (double*)calloc(hidden_size, sizeof(double));
    nn->w2 = (double**)malloc(output_size * sizeof(double*));
    nn->b2 = (double*)calloc(output_size, sizeof(double));

    nn->dw1 = (double**)malloc(hidden_size * sizeof(double*));
    nn->db1 = (double*)calloc(hidden_size, sizeof(double));
    nn->dw2 = (double**)malloc(output_size * sizeof(double*));
    nn->db2 = (double*)calloc(output_size, sizeof(double));

    // Allocate activations
    nn->hidden = (double*)calloc(hidden_size, sizeof(double));
    nn->output = (double*)calloc(output_size, sizeof(double));

    // Check allocations
    if (!nn->w1 || !nn->b1 || !nn->w2 || !nn->b2 ||
        !nn->dw1 || !nn->db1 || !nn->dw2 || !nn->db2 ||
        !nn->hidden || !nn->output) {
        perror("Memory allocation failed");
        free_network(nn);
        exit(EXIT_FAILURE);
    }

    // Initialize weights with random values
    for (int i = 0; i < hidden_size; i++) {
        nn->w1[i] = (double*)malloc(input_size * sizeof(double));
        nn->dw1[i] = (double*)calloc(input_size, sizeof(double));
        if (!nn->w1[i] || !nn->dw1[i]) {
            perror("Memory allocation failed");
            free_network(nn);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < input_size; j++) {
            nn->w1[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
        }
    }

    for (int i = 0; i < output_size; i++) {
        nn->w2[i] = (double*)malloc(hidden_size * sizeof(double));
        nn->dw2[i] = (double*)calloc(hidden_size, sizeof(double));
        if (!nn->w2[i] || !nn->dw2[i]) {
            perror("Memory allocation failed");
            free_network(nn);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < hidden_size; j++) {
            nn->w2[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
        }
    }

    return nn;
}

// Forward propagation
void forward(NeuralNetwork* nn, double* input) {

    // Hidden layer calculation
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->hidden[i] = 0;
        for (int j = 0; j < nn->input_size; j++) {
            nn->hidden[i] += input[j] * nn->w1[i][j];
        }
        nn->hidden[i] += nn->b1[i];
        nn->hidden[i] = sigmoid(nn->hidden[i]);
    }

    // Output layer calculation
    for (int i = 0; i < nn->output_size; i++) {
        nn->output[i] = 0;
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->output[i] += nn->hidden[j] * nn->w2[i][j];
        }
        nn->output[i] += nn->b2[i];
        nn->output[i] = sigmoid(nn->output[i]);
    }
}

// Backward propagation
void backward(NeuralNetwork* nn, double* input, double* target, double learning_rate) {

    // Calculate output layer error and gradients
    double* output_error = (double*)malloc(nn->output_size * sizeof(double));
    if (!output_error) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < nn->output_size; i++) {
        output_error[i] = target[i] - nn->output[i];

        // Calculate output layer gradients
        double delta_output = output_error[i] * sigmoid_derivative(nn->output[i]);

        // Update output layer weights
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->dw2[i][j] = delta_output * nn->hidden[j];
            nn->w2[i][j] += learning_rate * nn->dw2[i][j];
        }
        nn->db2[i] = delta_output;
        nn->b2[i] += learning_rate * nn->db2[i];
    }

    // Calculate hidden layer error and gradients
    for (int i = 0; i < nn->hidden_size; i++) {
        double hidden_error = 0;
        for (int j = 0; j < nn->output_size; j++) {
            hidden_error += output_error[j] * nn->w2[j][i];
        }

        double delta_hidden = hidden_error * sigmoid_derivative(nn->hidden[i]);

        // Update hidden layer weights
        for (int j = 0; j < nn->input_size; j++) {
            nn->dw1[i][j] = delta_hidden * input[j];
            nn->w1[i][j] += learning_rate * nn->dw1[i][j];
        }
        nn->db1[i] = delta_hidden;
        nn->b1[i] += learning_rate * nn->db1[i];
    }

    free(output_error);
}

// Calculate mean squared error
double calculate_mse(NeuralNetwork* nn, double** inputs, double** targets, int num_samples) {
    double total_error = 0;
    for (int s = 0; s < num_samples; s++) {
        forward(nn, inputs[s]);
        for (int i = 0; i < nn->output_size; i++) {
            double error = targets[s][i] - nn->output[i];
            total_error += error * error;
        }
    }
    return total_error / (num_samples * nn->output_size);
}

// Free allocated memory
void free_network(NeuralNetwork* nn) {
    if (!nn)
        return;

    for (int i = 0; i < nn->hidden_size; i++) {
        if (nn->w1)
            free(nn->w1[i]);
        if (nn->dw1)
            free(nn->dw1[i]);
    }
    if (nn->w1)
        free(nn->w1);
    if (nn->dw1)
        free(nn->dw1);
    if (nn->b1)
        free(nn->b1);
    if (nn->db1)
        free(nn->db1);

    for (int i = 0; i < nn->output_size; i++) {
        if (nn->w2)
            free(nn->w2[i]);
        if (nn->dw2)
            free(nn->dw2[i]);
    }
    if (nn->w2)
        free(nn->w2);
    if (nn->dw2)
        free(nn->dw2);
    if (nn->b2)
        free(nn->b2);
    if (nn->db2)
        free(nn->db2);

    if (nn->hidden)
        free(nn->hidden);
    if (nn->output)
        free(nn->output);
    free(nn);
}

// Print progress bar
void print_progress(int epoch, int total_epochs, double error) {
    int bar_width = 50;
    float progress = (float)epoch / total_epochs;
    int pos = bar_width * progress;

    printf("[");
    for (int i = 0; i < bar_width; i++) {
        if (i < pos)
            printf("=");
        else if (i == pos)
            printf(">");
        else
            printf(" ");
    }
    printf("] %d%% | Epoch: %d/%d | Error: %.6f\r",
           (int)(progress * 100), epoch, total_epochs, error);
    fflush(stdout);
}
