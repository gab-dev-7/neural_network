#include "dataset.h"
#include "neural_network.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// helper function to convert string to activationtype
ActivationType get_activation_type(const char* name) {
    if (strcmp(name, "sigmoid") == 0)
        return ACTIVATION_SIGMOID;
    if (strcmp(name, "relu") == 0)
        return ACTIVATION_RELU;
    if (strcmp(name, "leaky_relu") == 0)
        return ACTIVATION_LEAKY_RELU;
    if (strcmp(name, "tanh") == 0)
        return ACTIVATION_TANH;
    if (strcmp(name, "linear") == 0)
        return ACTIVATION_LINEAR;
    return ACTIVATION_RELU; // default
}

// helper function to convert string to losstype
LossType get_loss_type(const char* name) {
    if (strcmp(name, "mse") == 0)
        return LOSS_MSE;
    if (strcmp(name, "bce") == 0 || strcmp(name, "binary_cross_entropy") == 0)
        return LOSS_BINARY_CE;
    if (strcmp(name, "mae") == 0)
        return LOSS_MAE;
    return LOSS_MSE; // default
}

int main(int argc, char* argv[]) {
    // parse arguments
    TrainingConfig config = parse_arguments(argc, argv);

    if (config.verbose) {
        print_config(&config);
    }

    // create dataset based on configuration
    Dataset* full_dataset = NULL;
    Dataset* train_dataset = NULL;
    Dataset* test_dataset = NULL;

    if (strcmp(config.dataset_type, "sine") == 0) {
        full_dataset = create_sine_wave_dataset(config.dataset_size);
    } else if (strcmp(config.dataset_type, "circle") == 0) {
        full_dataset = create_circle_dataset(config.dataset_size);
    } else if (strcmp(config.dataset_type, "circle_enhanced") == 0) {
        full_dataset = create_circle_dataset_enhanced(config.dataset_size, config.boundary_ratio);
    } else if (strcmp(config.dataset_type, "xor") == 0) {
        full_dataset = create_xor_dataset();
    } else {
        printf("unknown dataset type: %s. using sine wave.\n", config.dataset_type);
        full_dataset = create_sine_wave_dataset(config.dataset_size);
    }

    // split into train and test sets
    split_dataset(full_dataset, config.train_test_split, &train_dataset, &test_dataset);

    if (config.verbose) {
        printf("\ndataset info:\n");
        printf("  type: %s\n", config.dataset_type);
        printf("  total samples: %d\n", full_dataset->num_samples);
        printf("  training samples: %d\n", train_dataset->num_samples);
        printf("  test samples: %d\n", test_dataset->num_samples);
        printf("  input size: %d\n", full_dataset->input_size);
        printf("  output size: %d\n", full_dataset->output_size);
        if (strcmp(config.dataset_type, "circle_enhanced") == 0) {
            printf("  boundary samples ratio: %.2f\n", config.boundary_ratio);
        }
    }

    // convert activation strings to types
    ActivationType hidden_act = get_activation_type(config.hidden_activation);
    ActivationType output_act = get_activation_type(config.output_activation);
    LossType loss_type = get_loss_type(config.loss_function);

    // create network
    NeuralNetwork* nn = create_network(full_dataset->input_size,
                                       config.hidden_size,
                                       full_dataset->output_size,
                                       hidden_act,
                                       output_act,
                                       loss_type);
    nn->momentum = config.momentum;
    nn->weight_decay = config.weight_decay;

    // gradient checking (if enabled)
    if (config.gradient_check) {
        printf("\n=== running gradient check ===\n");
        printf("testing with sample 0\n");
        double max_diff = gradient_check(nn, train_dataset->inputs[0],
                                         train_dataset->targets[0], 1e-4);

        printf("\ntesting with sample 1\n");
        max_diff = gradient_check(nn, train_dataset->inputs[1],
                                  train_dataset->targets[1], 1e-4);

        if (max_diff < 1e-7) {
            printf("\n✓ all gradient checks passed! backpropagation is correct.\n");
        } else {
            printf("\n✗ gradient checks failed! there's an issue with backpropagation.\n");
            printf("continuing training anyway...\n");
        }
        printf("\n");
    }

    // train
    if (config.verbose)
        printf("starting training...\n");

    double best_test_error = 1e9;
    int no_improvement_count = 0;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        // calculate current learning rate with exponential decay
        double current_learning_rate = config.learning_rate;
        if (config.decay_rate < 1.0) {
            int decay_cycles = epoch / config.decay_steps;
            current_learning_rate = config.learning_rate * pow(config.decay_rate, decay_cycles);

            if (config.verbose && epoch % config.decay_steps == 0 && epoch > 0) {
                printf("epoch %d: learning rate decayed to %.6f\n", epoch, current_learning_rate);
            }
        }

        // train for one epoch
        double train_error = train_epoch(nn, train_dataset->inputs, train_dataset->targets,
                                         train_dataset->num_samples, config.batch_size,
                                         current_learning_rate, config.shuffle);

        // calculate test error
        double test_error = calculate_loss(nn, test_dataset->inputs,
                                           test_dataset->targets, test_dataset->num_samples);

        // early stopping check
        if (test_error < best_test_error) {
            best_test_error = test_error;
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
            if (no_improvement_count >= config.patience && config.patience > 0) {
                if (config.verbose) {
                    printf("\nearly stopping at epoch %d (no improvement for %d epochs)\n",
                           epoch, config.patience);
                    printf("best test error: %.6f\n", best_test_error);
                }
                break;
            }
        }

        if (config.verbose && (epoch % (config.epochs / 20) == 0 || epoch == config.epochs - 1)) {
            printf("epoch %d/%d | train loss: %.6f | test loss: %.6f\n",
                   epoch, config.epochs, train_error, test_error);
        }

        // print progress bar in non-verbose mode
        if (!config.verbose && epoch % (config.epochs / 100) == 0) {
            print_progress(epoch, config.epochs, test_error);
        }
    }

    if (!config.verbose) {
        printf("\n"); // clear the progress bar line
    }

    if (config.verbose) {
        double final_train_error = calculate_loss(nn, train_dataset->inputs,
                                                  train_dataset->targets, train_dataset->num_samples);
        double final_test_error = calculate_loss(nn, test_dataset->inputs,
                                                 test_dataset->targets, test_dataset->num_samples);
        printf("\nfinal results:\n");
        printf("  training loss: %.6f\n", final_train_error);
        printf("  test loss: %.6f\n", final_test_error);

        // for classification problems, calculate accuracy
        if (loss_type == LOSS_BINARY_CE || strcmp(config.output_activation, "sigmoid") == 0) {
            int train_correct = 0, test_correct = 0;
            double threshold = 0.5;

            // training accuracy
            for (int i = 0; i < train_dataset->num_samples; i++) {
                forward(nn, train_dataset->inputs[i]);
                int predicted = (nn->output[0] > threshold) ? 1 : 0;
                int actual = (train_dataset->targets[i][0] > threshold) ? 1 : 0;
                if (predicted == actual)
                    train_correct++;
            }

            // test accuracy
            for (int i = 0; i < test_dataset->num_samples; i++) {
                forward(nn, test_dataset->inputs[i]);
                int predicted = (nn->output[0] > threshold) ? 1 : 0;
                int actual = (test_dataset->targets[i][0] > threshold) ? 1 : 0;
                if (predicted == actual)
                    test_correct++;
            }

            printf("  training accuracy: %.2f%% (%d/%d)\n",
                   100.0 * train_correct / train_dataset->num_samples,
                   train_correct, train_dataset->num_samples);
            printf("  test accuracy: %.2f%% (%d/%d)\n",
                   100.0 * test_correct / test_dataset->num_samples,
                   test_correct, test_dataset->num_samples);
        }
    }

    // test on a few examples
    printf("\nsample predictions:\n");
    int num_test_samples_to_show = test_dataset->num_samples < 10 ? test_dataset->num_samples : 10;
    for (int i = 0; i < num_test_samples_to_show; i++) {
        forward(nn, test_dataset->inputs[i]);
        printf("input: [");
        for (int j = 0; j < full_dataset->input_size; j++) {
            printf("%.3f", test_dataset->inputs[i][j]);
            if (j < full_dataset->input_size - 1)
                printf(", ");
        }
        printf("] -> output: %.4f (expected: %.4f)", nn->output[0], test_dataset->targets[i][0]);

        // show classification result for binary problems
        if (loss_type == LOSS_BINARY_CE || strcmp(config.output_activation, "sigmoid") == 0) {
            int predicted = (nn->output[0] > 0.5) ? 1 : 0;
            int actual = (test_dataset->targets[i][0] > 0.5) ? 1 : 0;
            printf(" -> class: %d (expected: %d) %s",
                   predicted, actual,
                   (predicted == actual) ? "✓" : "✗");
        }
        printf("\n");
    }

    // clean up
    free_network(nn);
    free_dataset(full_dataset);
    free_dataset(train_dataset);
    free_dataset(test_dataset);

    return 0;
}
