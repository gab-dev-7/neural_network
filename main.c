#include "neural_network.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    // parse arguments
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

    double d00[] = {0.0, 0.0}; double t0[] = {0.0};
    double d01[] = {0.0, 1.0}; double t1[] = {1.0};
    double d10[] = {1.0, 0.0}; double t1b[] = {1.0};
    double d11[] = {1.0, 1.0}; double t0b[] = {0.0};

    inputs_raw[0] = d00; targets_raw[0] = t0;
    inputs_raw[1] = d01; targets_raw[1] = t1;
    inputs_raw[2] = d10; targets_raw[2] = t1b;
    inputs_raw[3] = d11; targets_raw[3] = t0b;

    // create network
    NeuralNetwork* nn = create_network(2, config.hidden_size, 1);

    // train
    if (config.verbose) printf("starting training...\n");

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        
        // shuffle or iterate? simple iteration for now
        for (int i = 0; i < num_samples; i++) {
            forward(nn, inputs_raw[i]);
            backward(nn, inputs_raw[i], targets_raw[i], config.learning_rate);
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

    // test
    printf("\ntesting network:\n");
    for (int i = 0; i < num_samples; i++) {
        forward(nn, inputs_raw[i]);
        printf("input: [%.0f, %.0f] -> output: %.4f (expected: %.0f)\n",
               inputs_raw[i][0], inputs_raw[i][1], nn->output[0], targets_raw[i][0]);
    }

    free_network(nn);
    return 0;
}
