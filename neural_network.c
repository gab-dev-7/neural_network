#include "neural_network.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// sigmoid activation
double sigmoid(double x) {
    if (x >= 0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        double ex = exp(x);
        return ex / (1.0 + ex);
    }
}

// relu activation
double relu(double x) {
    return (x > 0) ? x : 0.0;
}

// leaky relu activation
double leaky_relu(double x) {
    return (x > 0) ? x : 0.01 * x;
}

// tanh activation
double tanh_activation(double x) {
    return tanh(x);
}

// linear activation
double linear(double x) {
    return x;
}

// sigmoid derivative (from output)
double sigmoid_derivative_from_output(double output) {
    return output * (1.0 - output);
}

// relu derivative
double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// leaky relu derivative
double leaky_relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.01;
}

// tanh derivative
double tanh_derivative(double x) {
    double th = tanh(x);
    return 1.0 - th * th;
}

// linear derivative
double linear_derivative(void) {
    return 1.0;
}

// get activation function based on type
double activate(double x, ActivationType type) {
    switch (type) {
        case ACTIVATION_SIGMOID:
            return sigmoid(x);
        case ACTIVATION_RELU:
            return relu(x);
        case ACTIVATION_LEAKY_RELU:
            return leaky_relu(x);
        case ACTIVATION_TANH:
            return tanh_activation(x);
        case ACTIVATION_LINEAR:
            return linear(x);
        default:
            return relu(x);
    }
}

// get activation derivative based on type
double activate_derivative(double x, ActivationType type) {
    switch (type) {
        case ACTIVATION_SIGMOID:
            return sigmoid_derivative_from_output(activate(x, ACTIVATION_SIGMOID));
        case ACTIVATION_RELU:
            return relu_derivative(x);
        case ACTIVATION_LEAKY_RELU:
            return leaky_relu_derivative(x);
        case ACTIVATION_TANH:
            return tanh_derivative(x);
        case ACTIVATION_LINEAR:
            return linear_derivative();
        default:
            return relu_derivative(x);
    }
}

// loss functions
double mse_loss(double output, double target) {
    double error = target - output;
    return error * error;
}

double binary_cross_entropy_loss(double output, double target) {
    // add epsilon to avoid log(0)
    double epsilon = 1e-12;
    output = fmax(epsilon, fmin(1.0 - epsilon, output));
    return -(target * log(output) + (1.0 - target) * log(1.0 - output));
}

double mae_loss(double output, double target) {
    return fabs(target - output);
}

double compute_loss(double output, double target, LossType type) {
    switch (type) {
        case LOSS_MSE:
            return mse_loss(output, target);
        case LOSS_BINARY_CE:
            return binary_cross_entropy_loss(output, target);
        case LOSS_MAE:
            return mae_loss(output, target);
        default:
            return mse_loss(output, target);
    }
}

double compute_loss_derivative(double output, double target, LossType type) {
    double epsilon = 1e-12;
    output = fmax(epsilon, fmin(1.0 - epsilon, output));

    switch (type) {
        case LOSS_MSE:
            return 2.0 * (output - target); // dL/doutput for mse
        case LOSS_BINARY_CE:
            // for bce: dL/doutput = (output - target) / (output * (1 - output))
            return (output - target) / (output * (1 - output));
        case LOSS_MAE:
            return (output > target) ? 1.0 : -1.0;
        default:
            return 2.0 * (output - target);
    }
}

// initialize neural network with activation types
NeuralNetwork* create_network(int input_size, int hidden_size, int output_size,
                              ActivationType hidden_activation, ActivationType output_activation,
                              LossType loss_type) {

    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        perror("failed to allocate neural network");
        exit(EXIT_FAILURE);
    }

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;
    nn->hidden_activation = hidden_activation;
    nn->output_activation = output_activation;
    nn->loss_type = loss_type;
    nn->weight_decay = 0.0;

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
    nn->momentum = 0.0;

    // allocate activations
    nn->hidden = (double*)calloc(hidden_size, sizeof(double));
    nn->output = (double*)calloc(output_size, sizeof(double));
    nn->z_hidden = (double*)calloc(hidden_size, sizeof(double));
    nn->z_output = (double*)calloc(output_size, sizeof(double));

    // check allocations
    if (!nn->w1 || !nn->b1 || !nn->w2 || !nn->b2 ||
        !nn->dw1 || !nn->db1 || !nn->dw2 || !nn->db2 ||
        !nn->prev_dw1 || !nn->prev_dw2 ||
        !nn->hidden || !nn->output || !nn->z_hidden || !nn->z_output) {
        perror("memory allocation failed");
        free_network(nn);
        exit(EXIT_FAILURE);
    }

    // initialize weights with appropriate initialization
    for (int i = 0; i < hidden_size; i++) {
        nn->w1[i] = (double*)malloc(input_size * sizeof(double));
        nn->dw1[i] = (double*)calloc(input_size, sizeof(double));
        nn->prev_dw1[i] = (double*)calloc(input_size, sizeof(double));
        if (!nn->w1[i] || !nn->dw1[i] || !nn->prev_dw1[i]) {
            perror("memory allocation failed");
            free_network(nn);
            exit(EXIT_FAILURE);
        }

        // he initialization for relu, xavier for others
        for (int j = 0; j < input_size; j++) {
            double limit;
            if (hidden_activation == ACTIVATION_RELU || hidden_activation == ACTIVATION_LEAKY_RELU) {
                limit = sqrt(2.0 / input_size); // he initialization
            } else {
                limit = sqrt(2.0 / (input_size + hidden_size)); // xavier initialization
            }
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
            double limit2;
            if (output_activation == ACTIVATION_RELU || output_activation == ACTIVATION_LEAKY_RELU) {
                limit2 = sqrt(2.0 / hidden_size); // he initialization
            } else {
                limit2 = sqrt(2.0 / (hidden_size + output_size)); // xavier initialization
            }
            nn->w2[i][j] = ((double)rand() / RAND_MAX) * 2 * limit2 - limit2;
        }
    }

    return nn;
}

// forward propagation with different activations
void forward(NeuralNetwork* nn, double* input) {
    // hidden layer calculation
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->z_hidden[i] = nn->b1[i]; // start with bias
        for (int j = 0; j < nn->input_size; j++) {
            nn->z_hidden[i] += input[j] * nn->w1[i][j];
        }
        nn->hidden[i] = activate(nn->z_hidden[i], nn->hidden_activation);
    }

    // output layer calculation
    for (int i = 0; i < nn->output_size; i++) {
        nn->z_output[i] = nn->b2[i]; // start with bias
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->z_output[i] += nn->hidden[j] * nn->w2[i][j];
        }
        nn->output[i] = activate(nn->z_output[i], nn->output_activation);
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
        // use appropriate loss derivative
        output_error[i] = compute_loss_derivative(nn->output[i], target[i], nn->loss_type);

        // calculate output layer gradients
        double delta_output = output_error[i] * activate_derivative(nn->z_output[i], nn->output_activation);

        // accumulate output layer gradients (no weight update)
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

        double delta_hidden = hidden_error * activate_derivative(nn->z_hidden[i], nn->hidden_activation);

        // accumulate hidden layer gradients (no weight update)
        for (int j = 0; j < nn->input_size; j++) {
            nn->dw1[i][j] += delta_hidden * input[j];
        }
        nn->db1[i] += delta_hidden;
    }

    free(output_error);
}

// update weights using accumulated gradients (called once per batch)
void update_weights(NeuralNetwork* nn, double learning_rate, int batch_size) {
    // update output layer weights with momentum and weight decay
    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            // average gradient over batch
            double avg_gradient = nn->dw2[i][j] / batch_size;

            // momentum update + weight decay
            double update = -learning_rate * avg_gradient + nn->momentum * nn->prev_dw2[i][j];

            // apply weight decay (l2 regularization)
            update -= learning_rate * nn->weight_decay * nn->w2[i][j];

            nn->w2[i][j] += update;
            nn->prev_dw2[i][j] = update; // store for next iteration
        }

        // average bias gradient over batch (no weight decay for biases)
        double avg_bias_gradient = nn->db2[i] / batch_size;
        double bias_update = -learning_rate * avg_bias_gradient;
        nn->b2[i] += bias_update;
    }

    // update hidden layer weights with momentum and weight decay
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            // average gradient over batch
            double avg_gradient = nn->dw1[i][j] / batch_size;

            // momentum update + weight decay
            double update = -learning_rate * avg_gradient + nn->momentum * nn->prev_dw1[i][j];

            // apply weight decay (l2 regularization)
            update -= learning_rate * nn->weight_decay * nn->w1[i][j];

            nn->w1[i][j] += update;
            nn->prev_dw1[i][j] = update; // store for next iteration
        }

        // average bias gradient over batch (no weight decay for biases)
        double avg_bias_gradient = nn->db1[i] / batch_size;
        double bias_update = -learning_rate * avg_bias_gradient;
        nn->b1[i] += bias_update;
    }
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

// train for one epoch with dataset
double train_epoch(NeuralNetwork* nn, double** inputs, double** targets,
                   int num_samples, int batch_size, double learning_rate, int shuffle) {

    // shuffle data if enabled
    if (shuffle) {
        // fisher-yates shuffle
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

    double total_loss = 0;

    // process in batches
    int batch_start = 0;
    while (batch_start < num_samples) {
        // calculate actual batch size (last batch might be smaller)
        int actual_batch_size = batch_size;
        if (batch_start + actual_batch_size > num_samples) {
            actual_batch_size = num_samples - batch_start;
        }

        // reset gradients for new batch
        reset_gradients(nn);

        // accumulate gradients over batch
        for (int i = 0; i < actual_batch_size; i++) {
            int sample_idx = batch_start + i;
            forward(nn, inputs[sample_idx]);
            backward_accumulate(nn, inputs[sample_idx], targets[sample_idx]);

            // calculate loss for this sample
            for (int j = 0; j < nn->output_size; j++) {
                total_loss += compute_loss(nn->output[j], targets[sample_idx][j], nn->loss_type);
            }
        }

        // update weights once per batch
        update_weights(nn, learning_rate, actual_batch_size);

        batch_start += actual_batch_size;
    }

    // return average loss per sample
    return total_loss / (num_samples * nn->output_size);
}

// calculate mean squared error or other loss
double calculate_loss(NeuralNetwork* nn, double** inputs, double** targets, int num_samples) {
    double total_loss = 0;
    for (int s = 0; s < num_samples; s++) {
        forward(nn, inputs[s]);
        for (int i = 0; i < nn->output_size; i++) {
            total_loss += compute_loss(nn->output[i], targets[s][i], nn->loss_type);
        }
    }
    return total_loss / (num_samples * nn->output_size);
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
    if (nn->z_hidden)
        free(nn->z_hidden);
    if (nn->z_output)
        free(nn->z_output);
    free(nn);
}

// gradient checking function
double gradient_check(NeuralNetwork* nn, double* input, double* target, double epsilon) {
    double max_diff = 0.0;
    double tolerance = 1e-7;

    printf("\n=== gradient checking ===\n");

    // make a backup of the network
    NeuralNetwork* nn_backup = create_network(nn->input_size, nn->hidden_size, nn->output_size,
                                              nn->hidden_activation, nn->output_activation,
                                              nn->loss_type);

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

    // first, compute the loss at the current point
    forward(nn, input);
    double original_loss = 0;
    for (int k = 0; k < nn->output_size; k++) {
        original_loss += compute_loss(nn->output[k], target[k], nn->loss_type);
    }

    // compute analytical gradients using backward_accumulate
    reset_gradients(nn);
    backward_accumulate(nn, input, target);

    printf("checking w1 gradients...\n");
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            double original = nn_backup->w1[i][j];

            // perturb w1[i][j] positively
            nn->w1[i][j] = original + epsilon;
            forward(nn, input);
            double loss_plus = 0;
            for (int k = 0; k < nn->output_size; k++) {
                loss_plus += compute_loss(nn->output[k], target[k], nn->loss_type);
            }

            // perturb w1[i][j] negatively
            nn->w1[i][j] = original - epsilon;
            forward(nn, input);
            double loss_minus = 0;
            for (int k = 0; k < nn->output_size; k++) {
                loss_minus += compute_loss(nn->output[k], target[k], nn->loss_type);
            }

            // numerical gradient: (f(x+ε) - f(x-ε)) / (2ε)
            double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);

            // restore original weight
            nn->w1[i][j] = original;

            // analytical gradient from backward_accumulate
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

void print_usage(const char* program_name) {
    printf("usage: %s [options]\n", program_name);
    printf("options:\n");
    printf("  -e <epochs>     number of training epochs (default: 10000)\n");
    printf("  -l <rate>       learning rate (default: 0.1)\n");
    printf("  -h <size>       hidden layer size (default: 8)\n");
    printf("  -b <size>       batch size (default: 32)\n");
    printf("  -m <momentum>   momentum (default: 0.9)\n");
    printf("  -dr <rate>      learning rate decay rate (default: 0.995)\n");
    printf("  -ds <steps>     decay steps (default: 100)\n");
    printf("  -d <type>       dataset type: xor, sine, circle, circle_enhanced (default: sine)\n");
    printf("  -n <size>       dataset size (default: 1000)\n");
    printf("  -split <ratio>  train/test split ratio (default: 0.8)\n");
    printf("  -ha <act>       hidden activation: sigmoid, relu, leaky_relu, tanh, linear (default: relu)\n");
    printf("  -oa <act>       output activation: sigmoid, relu, leaky_relu, tanh, linear (default: sigmoid)\n");
    printf("  -loss <type>    loss function: mse, bce, mae (default: mse)\n");
    printf("  -wd <decay>     weight decay (l2 regularization) (default: 0.0)\n");
    printf("  -br <ratio>     boundary ratio for enhanced circle dataset (default: 0.3)\n");
    printf("  -s              shuffle data each epoch (default: yes)\n");
    printf("  -ns             no shuffle\n");
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
    config.hidden_size = 8;
    config.batch_size = 32;
    config.verbose = 0;
    config.validation_split = 0.0;
    config.patience = 50;
    config.momentum = 0.9;
    config.gradient_check = 0;
    config.shuffle = 1;

    config.decay_rate = 0.995;
    config.decay_steps = 100;

    strcpy(config.dataset_type, "sine");
    config.dataset_size = 1000;
    config.train_test_split = 0.8;
    config.boundary_ratio = 0.3;

    // activation function defaults
    strcpy(config.hidden_activation, "relu");
    strcpy(config.output_activation, "sigmoid");

    // loss function default
    strcpy(config.loss_function, "mse");

    // regularization default
    config.weight_decay = 0.0;

    // advanced features defaults
    config.use_validation_set = 0;
    config.validation_ratio = 0.15;
    config.use_enhanced_circle = 0;

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
        } else if (strcmp(argv[i], "-dr") == 0 && i + 1 < argc) {
            config.decay_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "-ds") == 0 && i + 1 < argc) {
            config.decay_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            strncpy(config.dataset_type, argv[++i], 19);
            config.dataset_type[19] = '\0';
            if (strcmp(config.dataset_type, "circle_enhanced") == 0) {
                config.use_enhanced_circle = 1;
            }
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            config.dataset_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-split") == 0 && i + 1 < argc) {
            config.train_test_split = atof(argv[++i]);
        } else if (strcmp(argv[i], "-ha") == 0 && i + 1 < argc) {
            strncpy(config.hidden_activation, argv[++i], 19);
            config.hidden_activation[19] = '\0';
        } else if (strcmp(argv[i], "-oa") == 0 && i + 1 < argc) {
            strncpy(config.output_activation, argv[++i], 19);
            config.output_activation[19] = '\0';
        } else if (strcmp(argv[i], "-loss") == 0 && i + 1 < argc) {
            strncpy(config.loss_function, argv[++i], 19);
            config.loss_function[19] = '\0';
        } else if (strcmp(argv[i], "-wd") == 0 && i + 1 < argc) {
            config.weight_decay = atof(argv[++i]);
        } else if (strcmp(argv[i], "-br") == 0 && i + 1 < argc) {
            config.boundary_ratio = atof(argv[++i]);
        } else if (strcmp(argv[i], "-g") == 0) {
            config.gradient_check = 1;
        } else if (strcmp(argv[i], "-s") == 0) {
            config.shuffle = 1;
        } else if (strcmp(argv[i], "-ns") == 0) {
            config.shuffle = 0;
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
    printf("  weight decay: %.6f\n", config->weight_decay);
    printf("  shuffle: %s\n", config->shuffle ? "yes" : "no");
    printf("  decay rate: %.6f\n", config->decay_rate);
    printf("  decay steps: %d\n", config->decay_steps);
    printf("  dataset: %s\n", config->dataset_type);
    printf("  dataset size: %d\n", config->dataset_size);
    printf("  train/test split: %.2f\n", config->train_test_split);
    printf("  hidden activation: %s\n", config->hidden_activation);
    printf("  output activation: %s\n", config->output_activation);
    printf("  loss function: %s\n", config->loss_function);
    printf("  gradient check: %s\n", config->gradient_check ? "yes" : "no");
    printf("  verbose: %s\n", config->verbose ? "yes" : "no");
    if (strcmp(config->dataset_type, "circle_enhanced") == 0) {
        printf("  boundary ratio: %.2f\n", config->boundary_ratio);
    }
}
