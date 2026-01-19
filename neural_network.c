#include "neural_network.h"
#include <stdlib.h>
#include <string.h>

// sigmoid activation: 1/(1+e^-x)
double sigmoid(double x) {
    if (x >= 0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        double ex = exp(x);
        return ex / (1.0 + ex);
    }
}

// derivative of sigmoid using output value: output * (1 - output)
double sigmoid_derivative_from_output(double output) {
    return output * (1.0 - output);
}

// create and initialize neural network
NeuralNetwork* create_network(int input_size, int hidden_size, int output_size) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        perror("failed to allocate neural network");
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
        perror("memory allocation failed");
        free_network(nn);
        exit(EXIT_FAILURE);
    }

    // initialize weights with random values using xavier initialization
    for (int i = 0; i < hidden_size; i++) {
        nn->w1[i] = (double*)malloc(input_size * sizeof(double));
        nn->dw1[i] = (double*)calloc(input_size, sizeof(double));
        nn->prev_dw1[i] = (double*)calloc(input_size, sizeof(double));
        if (!nn->w1[i] || !nn->dw1[i] || !nn->prev_dw1[i]) {
            perror("memory allocation failed");
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
            perror("memory allocation failed");
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
    // hidden layer calculation: sum(input * w1) + b1, then sigmoid
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->hidden[i] = 0;
        for (int j = 0; j < nn->input_size; j++) {
            nn->hidden[i] += input[j] * nn->w1[i][j];
        }
        nn->hidden[i] += nn->b1[i];
        nn->hidden[i] = sigmoid(nn->hidden[i]);
    }

    // output layer calculation: sum(hidden * w2) + b2, then sigmoid
    for (int i = 0; i < nn->output_size; i++) {
        nn->output[i] = 0;
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->output[i] += nn->hidden[j] * nn->w2[i][j];
        }
        nn->output[i] += nn->b2[i];
        nn->output[i] = sigmoid(nn->output[i]);
    }
}

// backward propagation (per-sample, updates weights immediately)
void backward(NeuralNetwork* nn, double* input, double* target, double learning_rate) {
    // calculate output layer error and gradients
    double* output_error = (double*)malloc(nn->output_size * sizeof(double));
    if (!output_error) {
        perror("memory allocation failed");
        return;
    }

    for (int i = 0; i < nn->output_size; i++) {
        // gradient of mse loss: (output - target)
        output_error[i] = nn->output[i] - target[i];

        // calculate output layer gradients
        double delta_output = output_error[i] * sigmoid_derivative_from_output(nn->output[i]);

        // update output layer weights with momentum
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->dw2[i][j] = delta_output * nn->hidden[j];
            // gradient descent: w = w - learning_rate * gradient
            double update = -learning_rate * nn->dw2[i][j] + nn->momentum * nn->prev_dw2[i][j];
            nn->w2[i][j] += update;
            nn->prev_dw2[i][j] = update;
        }
        nn->db2[i] = delta_output;
        nn->b2[i] -= learning_rate * nn->db2[i];
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
            double update = -learning_rate * nn->dw1[i][j] + nn->momentum * nn->prev_dw1[i][j];
            nn->w1[i][j] += update;
            nn->prev_dw1[i][j] = update;
        }
        nn->db1[i] = delta_hidden;
        nn->b1[i] -= learning_rate * nn->db1[i];
    }

    free(output_error);
}

// reset accumulated gradients to zero
void reset_gradients(NeuralNetwork* nn) {
    // reset hidden layer gradients
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            nn->dw1[i][j] = 0.0;
        }
        nn->db1[i] = 0.0;
    }

    // reset output layer gradients
    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->dw2[i][j] = 0.0;
        }
        nn->db2[i] = 0.0;
    }
}

// accumulate gradients for one sample (no weight update)
void backward_accumulate(NeuralNetwork* nn, double* input, double* target) {
    // calculate output layer error and gradients
    double* output_error = (double*)malloc(nn->output_size * sizeof(double));
    if (!output_error) {
        perror("memory allocation failed");
        return;
    }

    for (int i = 0; i < nn->output_size; i++) {
        // gradient of mse loss: (output - target)
        output_error[i] = nn->output[i] - target[i];

        // calculate output layer gradients
        double delta_output = output_error[i] * sigmoid_derivative_from_output(nn->output[i]);

        // accumulate output layer gradients
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->dw2[i][j] += delta_output * nn->hidden[j];
        }
        nn->db2[i] += delta_output;
    }

    // calculate hidden layer error and gradients
    for (int i = 0; i < nn->hidden_size; i++) {
        double hidden_error = 0;
        for (int j = 0; j < nn->output_size; j++) {
            hidden_error += output_error[j] * nn->w2[j][i];
        }

        double delta_hidden = hidden_error * sigmoid_derivative_from_output(nn->hidden[i]);

        // accumulate hidden layer gradients
        for (int j = 0; j < nn->input_size; j++) {
            nn->dw1[i][j] += delta_hidden * input[j];
        }
        nn->db1[i] += delta_hidden;
    }

    free(output_error);
}

// update weights using accumulated gradients (called once per batch)
void update_weights(NeuralNetwork* nn, double learning_rate, int batch_size) {
    // update output layer weights with momentum
    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            // average gradient over batch
            double avg_gradient = nn->dw2[i][j] / batch_size;

            // momentum update: w = w - learning_rate * avg_gradient + momentum * previous_update
            double update = -learning_rate * avg_gradient + nn->momentum * nn->prev_dw2[i][j];
            nn->w2[i][j] += update;
            nn->prev_dw2[i][j] = update;
        }

        // average bias gradient over batch
        double avg_bias_gradient = nn->db2[i] / batch_size;
        double bias_update = -learning_rate * avg_bias_gradient;
        nn->b2[i] += bias_update;
    }

    // update hidden layer weights with momentum
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            // average gradient over batch
            double avg_gradient = nn->dw1[i][j] / batch_size;

            // momentum update
            double update = -learning_rate * avg_gradient + nn->momentum * nn->prev_dw1[i][j];
            nn->w1[i][j] += update;
            nn->prev_dw1[i][j] = update;
        }

        // average bias gradient over batch
        double avg_bias_gradient = nn->db1[i] / batch_size;
        double bias_update = -learning_rate * avg_bias_gradient;
        nn->b1[i] += bias_update;
    }
}

// shuffle data function
void shuffle_data(double** inputs, double** targets, int num_samples, int input_size, int output_size) {
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        // swap inputs
        double* temp_input = inputs[i];
        inputs[i] = inputs[j];
        inputs[j] = temp_input;

        // swap targets
        double* temp_target = targets[i];
        targets[i] = targets[j];
        targets[j] = temp_target;
    }
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

// gradient checking function
double gradient_check(NeuralNetwork* nn, double* input, double* target, double epsilon) {
    double max_diff = 0.0;
    double tolerance = 1e-7;

    printf("\n=== gradient checking ===\n");

    // make a backup of the network
    NeuralNetwork* nn_backup = create_network(nn->input_size, nn->hidden_size, nn->output_size);

    // copy all weights and biases
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            nn_backup->w1[i][j] = nn->w1[i][j];
        }
        nn_backup->b1[i] = nn->b1[i];
    }
    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn_backup->w2[i][j] = nn->w2[i][j];
        }
        nn_backup->b2[i] = nn->b2[i];
    }

    // disable momentum for clean gradient computation
    double saved_momentum = nn->momentum;
    nn->momentum = 0.0;

    // compute the loss at the current point (with 0.5 factor for mse derivative)
    forward(nn, input);
    double original_loss = 0;
    for (int k = 0; k < nn->output_size; k++) {
        double error = target[k] - nn->output[k];
        original_loss += 0.5 * error * error;
    }

    // compute analytical gradients using backward()
    // use learning_rate = 1.0 to get raw gradients
    backward(nn, input, target, 1.0);

    printf("checking w1 gradients...\n");
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            double original = nn_backup->w1[i][j];

            // perturb w1[i][j] positively
            nn->w1[i][j] = original + epsilon;
            forward(nn, input);
            double loss_plus = 0;
            for (int k = 0; k < nn->output_size; k++) {
                double error = target[k] - nn->output[k];
                loss_plus += 0.5 * error * error;
            }

            // perturb w1[i][j] negatively
            nn->w1[i][j] = original - epsilon;
            forward(nn, input);
            double loss_minus = 0;
            for (int k = 0; k < nn->output_size; k++) {
                double error = target[k] - nn->output[k];
                loss_minus += 0.5 * error * error;
            }

            // numerical gradient: (f(x+ε) - f(x-ε)) / (2ε)
            double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);

            // restore original weight
            nn->w1[i][j] = original;

            // analytical gradient from backward()
            double analytical_gradient = nn->dw1[i][j];

            double diff = fabs(numerical_gradient - analytical_gradient);
            double avg = (fabs(numerical_gradient) + fabs(analytical_gradient)) / 2.0;
            double relative_diff = (avg > 1e-10) ? diff / avg : 0.0;

            if (relative_diff > max_diff) {
                max_diff = relative_diff;
            }

            if (relative_diff > tolerance) {
                printf("  w1[%d][%d]: num=%.6e, ana=%.6e, diff=%.6e, rel=%.6e\n",
                       i, j, numerical_gradient, analytical_gradient, diff, relative_diff);
            }
        }
    }

    // check b1 gradients
    printf("checking b1 gradients...\n");
    for (int i = 0; i < nn->hidden_size; i++) {
        double original = nn_backup->b1[i];

        nn->b1[i] = original + epsilon;
        forward(nn, input);
        double loss_plus = 0;
        for (int k = 0; k < nn->output_size; k++) {
            double error = target[k] - nn->output[k];
            loss_plus += 0.5 * error * error;
        }

        nn->b1[i] = original - epsilon;
        forward(nn, input);
        double loss_minus = 0;
        for (int k = 0; k < nn->output_size; k++) {
            double error = target[k] - nn->output[k];
            loss_minus += 0.5 * error * error;
        }

        double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);
        nn->b1[i] = original;

        double analytical_gradient = nn->db1[i];

        double diff = fabs(numerical_gradient - analytical_gradient);
        double avg = (fabs(numerical_gradient) + fabs(analytical_gradient)) / 2.0;
        double relative_diff = (avg > 1e-10) ? diff / avg : 0.0;

        if (relative_diff > max_diff)
            max_diff = relative_diff;
        if (relative_diff > tolerance) {
            printf("  b1[%d]: num=%.6e, ana=%.6e, diff=%.6e, rel=%.6e\n",
                   i, numerical_gradient, analytical_gradient, diff, relative_diff);
        }
    }

    // check w2 gradients
    printf("checking w2 gradients...\n");
    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            double original = nn_backup->w2[i][j];

            nn->w2[i][j] = original + epsilon;
            forward(nn, input);
            double loss_plus = 0;
            for (int k = 0; k < nn->output_size; k++) {
                double error = target[k] - nn->output[k];
                loss_plus += 0.5 * error * error;
            }

            nn->w2[i][j] = original - epsilon;
            forward(nn, input);
            double loss_minus = 0;
            for (int k = 0; k < nn->output_size; k++) {
                double error = target[k] - nn->output[k];
                loss_minus += 0.5 * error * error;
            }

            double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);
            nn->w2[i][j] = original;

            double analytical_gradient = nn->dw2[i][j];

            double diff = fabs(numerical_gradient - analytical_gradient);
            double avg = (fabs(numerical_gradient) + fabs(analytical_gradient)) / 2.0;
            double relative_diff = (avg > 1e-10) ? diff / avg : 0.0;

            if (relative_diff > max_diff)
                max_diff = relative_diff;
            if (relative_diff > tolerance) {
                printf("  w2[%d][%d]: num=%.6e, ana=%.6e, diff=%.6e, rel=%.6e\n",
                       i, j, numerical_gradient, analytical_gradient, diff, relative_diff);
            }
        }
    }

    // check b2 gradients
    printf("checking b2 gradients...\n");
    for (int i = 0; i < nn->output_size; i++) {
        double original = nn_backup->b2[i];

        nn->b2[i] = original + epsilon;
        forward(nn, input);
        double loss_plus = 0;
        for (int k = 0; k < nn->output_size; k++) {
            double error = target[k] - nn->output[k];
            loss_plus += 0.5 * error * error;
        }

        nn->b2[i] = original - epsilon;
        forward(nn, input);
        double loss_minus = 0;
        for (int k = 0; k < nn->output_size; k++) {
            double error = target[k] - nn->output[k];
            loss_minus += 0.5 * error * error;
        }

        double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);
        nn->b2[i] = original;

        double analytical_gradient = nn->db2[i];

        double diff = fabs(numerical_gradient - analytical_gradient);
        double avg = (fabs(numerical_gradient) + fabs(analytical_gradient)) / 2.0;
        double relative_diff = (avg > 1e-10) ? diff / avg : 0.0;

        if (relative_diff > max_diff)
            max_diff = relative_diff;
        if (relative_diff > tolerance) {
            printf("  b2[%d]: num=%.6e, ana=%.6e, diff=%.6e, rel=%.6e\n",
                   i, numerical_gradient, analytical_gradient, diff, relative_diff);
        }
    }

    // restore network
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            nn->w1[i][j] = nn_backup->w1[i][j];
        }
        nn->b1[i] = nn_backup->b1[i];
    }
    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->w2[i][j] = nn_backup->w2[i][j];
        }
        nn->b2[i] = nn_backup->b2[i];
    }

    // restore momentum
    nn->momentum = saved_momentum;

    // free backup
    free_network(nn_backup);

    printf("\nmaximum relative difference: %.2e\n", max_diff);
    if (max_diff < 1e-4) {
        printf("✓ gradient check passed!\n");
    } else {
        printf("✗ gradient check failed!\n");
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
    printf("] %d%% | epoch: %d/%d | error: %.6f\r",
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
    printf("  -b <size>       batch size (default: 1)\n");
    printf("  -m <momentum>   momentum (default: 0.0)\n");
    printf("  -g              enable gradient checking (before training)\n");
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
    config.validation_split = 0.0;
    config.patience = 10;
    config.momentum = 0.0;
    config.gradient_check = 0;
    config.shuffle = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            config.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            config.learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            config.hidden_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            config.batch_size = atoi(argv[++i]);
            if (config.batch_size <= 0)
                config.batch_size = 1;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            config.momentum = atof(argv[++i]);
        } else if (strcmp(argv[i], "-g") == 0) {
            config.gradient_check = 1;
        } else if (strcmp(argv[i], "-s") == 0) {
            config.shuffle = 1;
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
    printf("  batch size: %d\n", config->batch_size);
    printf("  momentum: %.6f\n", config->momentum);
    printf("  shuffle: %s\n", config->shuffle ? "yes" : "no");
    printf("  gradient check: %s\n", config->gradient_check ? "yes" : "no");
    printf("  verbose: %s\n", config->verbose ? "yes" : "no");
}
