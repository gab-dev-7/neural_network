#include "neural_network.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
    // parse command line arguments
    TrainingConfig config = parse_arguments(argc, argv);

    if (config.verbose) {
        print_config(&config);
    }

    // xor problem inputs and targets
    int num_samples = 4;
    double* inputs_raw[4];
    double* targets_raw[4];

    // xor logic table
    // 0,0 -> 0
    // 0,1 -> 1
    // 1,0 -> 1
    // 1,1 -> 0

    double d00[] = {0.0, 0.0};
    double t0[] = {0.0};
    double d01[] = {0.0, 1.0};
    double t1[] = {1.0};
    double d10[] = {1.0, 0.0};
    double t1b[] = {1.0};
    double d11[] = {1.0, 1.0};
    double t0b[] = {0.0};

    inputs_raw[0] = d00;
    targets_raw[0] = t0;
    inputs_raw[1] = d01;
    targets_raw[1] = t1;
    inputs_raw[2] = d10;
    targets_raw[2] = t1b;
    inputs_raw[3] = d11;
    targets_raw[3] = t0b;

    // create network with 2 inputs, hidden_size neurons, 1 output
    NeuralNetwork* nn = create_network(2, config.hidden_size, 1);
    nn->momentum = config.momentum;

    // gradient checking (if enabled)
    if (config.gradient_check) {
        printf("\n=== running gradient check ===\n");
        printf("testing with sample 0: input [0, 0], target [0]\n");
        double max_diff = gradient_check(nn, inputs_raw[0], targets_raw[0], 1e-4);

        printf("\ntesting with sample 1: input [0, 1], target [1]\n");
        max_diff = gradient_check(nn, inputs_raw[1], targets_raw[1], 1e-4);

        if (max_diff < 1e-7) {
            printf("\n✓ all gradient checks passed! backpropagation is correct.\n");
        } else {
            printf("\n✗ gradient checks failed! there's an issue with backpropagation.\n");
            printf("continuing training anyway...\n");
        }
        printf("\n");
    }

    // training loop
    if (config.verbose)
        printf("starting training...\n");

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        // shuffle data each epoch if enabled
        if (config.shuffle) {
            shuffle_data(inputs_raw, targets_raw, num_samples, 2, 1);
        }

        // process in batches
        int batch_start = 0;
        while (batch_start < num_samples) {
            // calculate actual batch size (last batch might be smaller)
            int actual_batch_size = config.batch_size;
            if (batch_start + actual_batch_size > num_samples) {
                actual_batch_size = num_samples - batch_start;
            }

            // reset gradients for new batch
            reset_gradients(nn);

            // accumulate gradients over batch
            for (int i = 0; i < actual_batch_size; i++) {
                int sample_idx = batch_start + i;
                forward(nn, inputs_raw[sample_idx]);
                backward_accumulate(nn, inputs_raw[sample_idx], targets_raw[sample_idx]);
            }

            // update weights once per batch
            update_weights(nn, config.learning_rate, actual_batch_size);

            batch_start += actual_batch_size;
        }

        if (config.verbose && (epoch % (config.epochs / 20) == 0)) {
            double mse = calculate_mse(nn, inputs_raw, targets_raw, num_samples);
            print_progress(epoch, config.epochs, mse);
        }
    }

    if (config.verbose) {
        double final_mse = calculate_mse(nn, inputs_raw, targets_raw, num_samples);
        print_progress(config.epochs, config.epochs, final_mse);
        printf("\n");
    }

    // test the trained network
    printf("\ntesting network:\n");
    for (int i = 0; i < num_samples; i++) {
        forward(nn, inputs_raw[i]);
        printf("input: [%.0f, %.0f] -> output: %.4f (expected: %.0f)\n",
               inputs_raw[i][0], inputs_raw[i][1], nn->output[0], targets_raw[i][0]);
    }

    free_network(nn);
    return 0;
}
