#include "neural_network.h"
#include <stdlib.h>
#include <string.h>

// activation function
double sigmoid(double x) {
    if (x >= 0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        double ex = exp(x);
        return ex / (1.0 + ex);
    }
}

// derivative of sigmoid
double sigmoid_derivative_from_output(double output) {
    return output * (1.0 - output);
}

// initialize neural network
NeuralNetwork* create_network(int input_size, int hidden_size, int output_size) {

    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        perror("Failed to allocate neural network");
        exit(EXIT_FAILURE);
    }

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    // seed random number generator
    srand(time(NULL));

    // allocate memory for weights and biases
    nn->w1 = (double**)malloc(hidden_size * sizeof(double*));
    nn->b1 = (double*)calloc(hidden_size, sizeof(double));
    nn->w2 = (double**)malloc(output_size * sizeof(double*));
    nn->b2 = (double*)calloc(output_size, sizeof(double));

    nn->dw1 = (double**)malloc(hidden_size * sizeof(double*));
    nn->db1 = (double*)calloc(hidden_size, sizeof(double));
    nn->dw2 = (double**)malloc(output_size * sizeof(double*));
    nn->db2 = (double*)calloc(output_size, sizeof(double));

    // allocate memory for momentum buffers
    nn->prev_dw1 = (double**)malloc(hidden_size * sizeof(double*));
    nn->prev_dw2 = (double**)malloc(output_size * sizeof(double*));
    nn->momentum = 0.0; // default momentum

    // allocate activations
    nn->hidden = (double*)calloc(hidden_size, sizeof(double));
    nn->output = (double*)calloc(output_size, sizeof(double));

    // check allocations
    if (!nn->w1 || !nn->b1 || !nn->w2 || !nn->b2 ||
        !nn->dw1 || !nn->db1 || !nn->dw2 || !nn->db2 ||
        !nn->prev_dw1 || !nn->prev_dw2 ||
        !nn->hidden || !nn->output) {
        perror("Memory allocation failed");
        free_network(nn);
        exit(EXIT_FAILURE);
    }

    // initialize weights with random values
    for (int i = 0; i < hidden_size; i++) {
        nn->w1[i] = (double*)malloc(input_size * sizeof(double));
        nn->dw1[i] = (double*)calloc(input_size, sizeof(double));
        nn->prev_dw1[i] = (double*)calloc(input_size, sizeof(double));
        if (!nn->w1[i] || !nn->dw1[i] || !nn->prev_dw1[i]) {
            perror("Memory allocation failed");
            free_network(nn);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < input_size; j++) {
            double limit = sqrt(6.0 / (input_size + hidden_size));
            nn->w1[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
        }
    }

    for (int i = 0; i < output_size; i++) {
        nn->w2[i] = (double*)malloc(hidden_size * sizeof(double));
        nn->dw2[i] = (double*)calloc(hidden_size, sizeof(double));
        nn->prev_dw2[i] = (double*)calloc(hidden_size, sizeof(double));
        if (!nn->w2[i] || !nn->dw2[i] || !nn->prev_dw2[i]) {
            perror("Memory allocation failed");
            free_network(nn);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < hidden_size; j++) {
            double limit2 = sqrt(6.0 / (hidden_size + output_size));
            nn->w2[i][j] = ((double)rand() / RAND_MAX) * 2 * limit2 - limit2;
        }
    }

    return nn;
}

// forward propagation
void forward(NeuralNetwork* nn, double* input) {

    // hidden layer calculation
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->hidden[i] = 0;
        for (int j = 0; j < nn->input_size; j++) {
            nn->hidden[i] += input[j] * nn->w1[i][j];
        }
        nn->hidden[i] += nn->b1[i];
        nn->hidden[i] = sigmoid(nn->hidden[i]);
    }

    // output layer calculation
    for (int i = 0; i < nn->output_size; i++) {
        nn->output[i] = 0;
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->output[i] += nn->hidden[j] * nn->w2[i][j];
        }
        nn->output[i] += nn->b2[i];
        nn->output[i] = sigmoid(nn->output[i]);
    }
}

// backward propagation
void backward(NeuralNetwork* nn, double* input, double* target, double learning_rate) {

    // calculate output layer error and gradients
    double* output_error = (double*)malloc(nn->output_size * sizeof(double));
    if (!output_error) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < nn->output_size; i++) {
        output_error[i] = target[i] - nn->output[i];

        // calculate output layer gradients
        double delta_output = output_error[i] * sigmoid_derivative_from_output(nn->output[i]);

        // update output layer weights with momentum
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->dw2[i][j] = delta_output * nn->hidden[j];
            // momentum update: current_gradient + momentum * previous_update
            double update = learning_rate * nn->dw2[i][j] + nn->momentum * nn->prev_dw2[i][j];
            nn->w2[i][j] += update;
            nn->prev_dw2[i][j] = update; // store for next iteration
        }
        nn->db2[i] = delta_output;
        nn->b2[i] += learning_rate * nn->db2[i];
        // Note: biases typically don't get momentum, but we could add if needed
    }

    // calculate hidden layer error and gradients
    for (int i = 0; i < nn->hidden_size; i++) {
        double hidden_error = 0;
        for (int j = 0; j < nn->output_size; j++) {
            hidden_error += output_error[j] * nn->w2[j][i];
        }

        double delta_hidden = hidden_error * sigmoid_derivative_from_output(nn->hidden[i]);

        // update hidden layer weights with momentum
        for (int j = 0; j < nn->input_size; j++) {
            nn->dw1[i][j] = delta_hidden * input[j];
            double update = learning_rate * nn->dw1[i][j] + nn->momentum * nn->prev_dw1[i][j];
            nn->w1[i][j] += update;
            nn->prev_dw1[i][j] = update; // store for next iteration
        }
        nn->db1[i] = delta_hidden;
        nn->b1[i] += learning_rate * nn->db1[i];
    }

    free(output_error);
}

// calculate mean squared error
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

// free allocated memory
void free_network(NeuralNetwork* nn) {
    if (!nn)
        return;

    for (int i = 0; i < nn->hidden_size; i++) {
        if (nn->w1)
            free(nn->w1[i]);
        if (nn->dw1)
            free(nn->dw1[i]);
        if (nn->prev_dw1)
            free(nn->prev_dw1[i]);
    }
    if (nn->w1)
        free(nn->w1);
    if (nn->dw1)
        free(nn->dw1);
    if (nn->prev_dw1)
        free(nn->prev_dw1);
    if (nn->b1)
        free(nn->b1);
    if (nn->db1)
        free(nn->db1);

    for (int i = 0; i < nn->output_size; i++) {
        if (nn->w2)
            free(nn->w2[i]);
        if (nn->dw2)
            free(nn->dw2[i]);
        if (nn->prev_dw2)
            free(nn->prev_dw2[i]);
    }
    if (nn->w2)
        free(nn->w2);
    if (nn->dw2)
        free(nn->dw2);
    if (nn->prev_dw2)
        free(nn->prev_dw2);
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

// NEW: Gradient checking function
double gradient_check(NeuralNetwork* nn, double* input, double* target, double epsilon) {
    double max_diff = 0.0;
    double tolerance = 1e-7;

    printf("\n=== Gradient Checking ===\n");

    // Check w1 gradients
    printf("Checking w1 gradients...\n");
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            // Save original weight
            double original = nn->w1[i][j];

            // Compute cost with w1[i][j] + epsilon
            nn->w1[i][j] = original + epsilon;
            forward(nn, input);
            double cost_plus = 0;
            for (int k = 0; k < nn->output_size; k++) {
                double error = target[k] - nn->output[k];
                cost_plus += 0.5 * error * error;
            }

            // Compute cost with w1[i][j] - epsilon
            nn->w1[i][j] = original - epsilon;
            forward(nn, input);
            double cost_minus = 0;
            for (int k = 0; k < nn->output_size; k++) {
                double error = target[k] - nn->output[k];
                cost_minus += 0.5 * error * error;
            }

            // Numerical gradient
            double numerical_gradient = (cost_plus - cost_minus) / (2 * epsilon);

            // Restore original weight
            nn->w1[i][j] = original;

            // Do a forward and backward to get analytical gradient
            forward(nn, input);

            // We need to call backward but without momentum for clean comparison
            // Temporarily disable momentum
            double saved_momentum = nn->momentum;
            nn->momentum = 0.0;

            // Clear gradients first
            for (int k = 0; k < nn->hidden_size; k++) {
                for (int l = 0; l < nn->input_size; l++) {
                    nn->dw1[k][l] = 0;
                }
            }

            // Calculate analytical gradient
            double* output_error = malloc(nn->output_size * sizeof(double));
            for (int k = 0; k < nn->output_size; k++) {
                output_error[k] = target[k] - nn->output[k];
                double delta_output = output_error[k] * sigmoid_derivative_from_output(nn->output[k]);

                for (int l = 0; l < nn->hidden_size; l++) {
                    nn->dw2[k][l] = delta_output * nn->hidden[l];
                }
            }

            // Calculate hidden layer error
            for (int k = 0; k < nn->hidden_size; k++) {
                double hidden_error = 0;
                for (int l = 0; l < nn->output_size; l++) {
                    hidden_error += output_error[l] * nn->w2[l][k];
                }
                double delta_hidden = hidden_error * sigmoid_derivative_from_output(nn->hidden[k]);

                for (int l = 0; l < nn->input_size; l++) {
                    nn->dw1[k][l] = delta_hidden * input[l];
                }
            }
            free(output_error);

            double analytical_gradient = nn->dw1[i][j];

            // Restore momentum
            nn->momentum = saved_momentum;

            // Calculate relative difference
            double diff = fabs(numerical_gradient - analytical_gradient);
            double relative_diff = diff / (fabs(numerical_gradient) + fabs(analytical_gradient) + 1e-10);

            if (relative_diff > max_diff) {
                max_diff = relative_diff;
            }

            if (relative_diff > tolerance) {
                printf("  w1[%d][%d]: numerical=%.10e, analytical=%.10e, diff=%.10e, relative=%.10e\n",
                       i, j, numerical_gradient, analytical_gradient, diff, relative_diff);
            }
        }
    }

    // Check b1 gradients
    printf("Checking b1 gradients...\n");
    for (int i = 0; i < nn->hidden_size; i++) {
        // Save original bias
        double original = nn->b1[i];

        // Compute cost with b1[i] + epsilon
        nn->b1[i] = original + epsilon;
        forward(nn, input);
        double cost_plus = 0;
        for (int k = 0; k < nn->output_size; k++) {
            double error = target[k] - nn->output[k];
            cost_plus += 0.5 * error * error;
        }

        // Compute cost with b1[i] - epsilon
        nn->b1[i] = original - epsilon;
        forward(nn, input);
        double cost_minus = 0;
        for (int k = 0; k < nn->output_size; k++) {
            double error = target[k] - nn->output[k];
            cost_minus += 0.5 * error * error;
        }

        // Numerical gradient
        double numerical_gradient = (cost_plus - cost_minus) / (2 * epsilon);

        // Restore original bias
        nn->b1[i] = original;

        // Analytical gradient (we already computed it above)
        double analytical_gradient = nn->db1[i];

        // Calculate relative difference
        double diff = fabs(numerical_gradient - analytical_gradient);
        double relative_diff = diff / (fabs(numerical_gradient) + fabs(analytical_gradient) + 1e-10);

        if (relative_diff > max_diff) {
            max_diff = relative_diff;
        }

        if (relative_diff > tolerance) {
            printf("  b1[%d]: numerical=%.10e, analytical=%.10e, diff=%.10e, relative=%.10e\n",
                   i, numerical_gradient, analytical_gradient, diff, relative_diff);
        }
    }

    printf("Maximum relative difference: %.10e\n", max_diff);
    if (max_diff < 1e-7) {
        printf("✓ Gradient check PASSED! (max diff < 1e-7)\n");
    } else {
        printf("✗ Gradient check FAILED! (max diff >= 1e-7)\n");
    }

    return max_diff;
}

// print progress bar
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

// print usage information
void print_usage(const char* program_name) {
    printf("usage: %s [options]\n", program_name);
    printf("options:\n");
    printf("  -e <epochs>     number of training epochs (default: 10000)\n");
    printf("  -l <rate>       learning rate (default: 0.1)\n");
    printf("  -h <size>       hidden layer size (default: 2)\n");
    printf("  -b <size>       batch size (default: 1) [ignored in this implementation]\n");
    printf("  -m <momentum>   momentum (default: 0.0)\n");
    printf("  -g              enable gradient checking (before training)\n"); // NEW
    printf("  -v              verbose mode\n");
    printf("  -?              show this help message\n");
}

// parse command line arguments
TrainingConfig parse_arguments(int argc, char* argv[]) {
    TrainingConfig config;
    // defaults
    config.epochs = 10000;
    config.learning_rate = 0.1;
    config.hidden_size = 2;
    config.batch_size = 1;
    config.verbose = 0;
    config.validation_split = 0.0; // Not implemented yet
    config.patience = 10;          // Not implemented yet
    config.momentum = 0.0;         // Default momentum

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            config.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            config.learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            config.hidden_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            config.batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            config.momentum = atof(argv[++i]);
        } else if (strcmp(argv[i], "-g") == 0) { // NEW: gradient checking flag
            config.gradient_check = 1;
        } else if (strcmp(argv[i], "-v") == 0) {
            config.verbose = 1;
        } else if (strcmp(argv[i], "-?") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }
    return config;
}

// print configuration
void print_config(const TrainingConfig* config) {
    printf("training configuration:\n");
    printf("  epochs: %d\n", config->epochs);
    printf("  learning rate: %.6f\n", config->learning_rate);
    printf("  hidden size: %d\n", config->hidden_size);
    printf("  momentum: %.6f\n", config->momentum);
    printf("  gradient check: %s\n", config->gradient_check ? "yes" : "no");
    printf("  verbose: %s\n", config->verbose ? "yes" : "no");
}
