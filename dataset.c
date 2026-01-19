#include "dataset.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// create sine wave dataset: z = sin(x) * cos(y)
Dataset* create_sine_wave_dataset(int num_samples) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->num_samples = num_samples;
    dataset->input_size = 2;
    dataset->output_size = 1;

    dataset->inputs = (double**)malloc(num_samples * sizeof(double*));
    dataset->targets = (double**)malloc(num_samples * sizeof(double*));

    srand(time(NULL));

    for (int i = 0; i < num_samples; i++) {
        dataset->inputs[i] = (double*)malloc(2 * sizeof(double));
        dataset->targets[i] = (double*)malloc(1 * sizeof(double));

        // generate random points in [-π, π]
        double x = (rand() / (double)RAND_MAX) * 2 * M_PI - M_PI;
        double y = (rand() / (double)RAND_MAX) * 2 * M_PI - M_PI;

        dataset->inputs[i][0] = x;
        dataset->inputs[i][1] = y;

        // target: sin(x) * cos(y), normalized to [0, 1]
        double z = sin(x) * cos(y);
        dataset->targets[i][0] = (z + 1.0) / 2.0;
    }

    return dataset;
}

// create xor dataset with 100 samples and noise
Dataset* create_xor_dataset() {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->num_samples = 100;
    dataset->input_size = 2;
    dataset->output_size = 1;

    dataset->inputs = (double**)malloc(dataset->num_samples * sizeof(double*));
    dataset->targets = (double**)malloc(dataset->num_samples * sizeof(double*));

    srand(time(NULL));

    for (int i = 0; i < dataset->num_samples; i++) {
        dataset->inputs[i] = (double*)malloc(2 * sizeof(double));
        dataset->targets[i] = (double*)malloc(1 * sizeof(double));

        // original xor patterns with small random noise
        int pattern = i % 4;
        double noise1 = ((rand() / (double)RAND_MAX) * 0.2 - 0.1);
        double noise2 = ((rand() / (double)RAND_MAX) * 0.2 - 0.1);

        switch (pattern) {
            case 0: // 0,0 -> 0
                dataset->inputs[i][0] = 0.0 + noise1;
                dataset->inputs[i][1] = 0.0 + noise2;
                dataset->targets[i][0] = 0.0;
                break;
            case 1: // 0,1 -> 1
                dataset->inputs[i][0] = 0.0 + noise1;
                dataset->inputs[i][1] = 1.0 + noise2;
                dataset->targets[i][0] = 1.0;
                break;
            case 2: // 1,0 -> 1
                dataset->inputs[i][0] = 1.0 + noise1;
                dataset->inputs[i][1] = 0.0 + noise2;
                dataset->targets[i][0] = 1.0;
                break;
            case 3: // 1,1 -> 0
                dataset->inputs[i][0] = 1.0 + noise1;
                dataset->inputs[i][1] = 1.0 + noise2;
                dataset->targets[i][0] = 0.0;
                break;
        }
    }

    return dataset;
}

// create a circle dataset (non-linearly separable)
Dataset* create_circle_dataset(int num_samples) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->num_samples = num_samples;
    dataset->input_size = 2;
    dataset->output_size = 1;

    dataset->inputs = (double**)malloc(num_samples * sizeof(double*));
    dataset->targets = (double**)malloc(num_samples * sizeof(double*));

    srand(time(NULL));

    for (int i = 0; i < num_samples; i++) {
        dataset->inputs[i] = (double*)malloc(2 * sizeof(double));
        dataset->targets[i] = (double*)malloc(1 * sizeof(double));

        // generate random point in [-1, 1] x [-1, 1]
        double x = (rand() / (double)RAND_MAX) * 2 - 1;
        double y = (rand() / (double)RAND_MAX) * 2 - 1;

        dataset->inputs[i][0] = x;
        dataset->inputs[i][1] = y;

        // classify as 1 if inside circle of radius 0.7, else 0
        double distance = sqrt(x * x + y * y);
        if (distance < 0.7) {
            dataset->targets[i][0] = 1.0;
        } else {
            dataset->targets[i][0] = 0.0;
        }
    }

    return dataset;
}

// create circle dataset with more points near the decision boundary
Dataset* create_circle_dataset_enhanced(int num_samples, double boundary_ratio) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->num_samples = num_samples;
    dataset->input_size = 2;
    dataset->output_size = 1;

    dataset->inputs = (double**)malloc(num_samples * sizeof(double*));
    dataset->targets = (double**)malloc(num_samples * sizeof(double*));

    srand(time(NULL));

    int boundary_samples = (int)(num_samples * boundary_ratio);
    int uniform_samples = num_samples - boundary_samples;

    double radius = 0.7;
    double boundary_width = 0.15; // points within radius ± boundary_width

    for (int i = 0; i < num_samples; i++) {
        dataset->inputs[i] = (double*)malloc(2 * sizeof(double));
        dataset->targets[i] = (double*)malloc(1 * sizeof(double));

        double x, y, distance;

        if (i < uniform_samples) {
            // uniform random samples
            x = (rand() / (double)RAND_MAX) * 2 - 1;
            y = (rand() / (double)RAND_MAX) * 2 - 1;
        } else {
            // samples near the decision boundary
            double angle = (rand() / (double)RAND_MAX) * 2 * M_PI;
            // generate radius near the decision boundary
            double r = radius + ((rand() / (double)RAND_MAX) * 2 - 1) * boundary_width;
            x = r * cos(angle);
            y = r * sin(angle);
        }

        dataset->inputs[i][0] = x;
        dataset->inputs[i][1] = y;

        // classify as 1 if inside circle of radius 0.7, else 0
        distance = sqrt(x * x + y * y);
        if (distance < radius) {
            dataset->targets[i][0] = 1.0;
        } else {
            dataset->targets[i][0] = 0.0;
        }
    }

    return dataset;
}

// split dataset into train and test sets
void split_dataset(Dataset* dataset, double train_ratio,
                   Dataset** train_set, Dataset** test_set) {

    if (train_ratio < 0 || train_ratio > 1)
        train_ratio = 0.8;

    int train_size = (int)(dataset->num_samples * train_ratio);
    int test_size = dataset->num_samples - train_size;

    // create train set
    *train_set = (Dataset*)malloc(sizeof(Dataset));
    (*train_set)->num_samples = train_size;
    (*train_set)->input_size = dataset->input_size;
    (*train_set)->output_size = dataset->output_size;
    (*train_set)->inputs = (double**)malloc(train_size * sizeof(double*));
    (*train_set)->targets = (double**)malloc(train_size * sizeof(double*));

    // create test set
    *test_set = (Dataset*)malloc(sizeof(Dataset));
    (*test_set)->num_samples = test_size;
    (*test_set)->input_size = dataset->input_size;
    (*test_set)->output_size = dataset->output_size;
    (*test_set)->inputs = (double**)malloc(test_size * sizeof(double*));
    (*test_set)->targets = (double**)malloc(test_size * sizeof(double*));

    // shuffle indices
    int* indices = (int*)malloc(dataset->num_samples * sizeof(int));
    for (int i = 0; i < dataset->num_samples; i++)
        indices[i] = i;

    // fisher-yates shuffle
    for (int i = dataset->num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // copy data to train set
    for (int i = 0; i < train_size; i++) {
        int idx = indices[i];
        (*train_set)->inputs[i] = (double*)malloc(dataset->input_size * sizeof(double));
        (*train_set)->targets[i] = (double*)malloc(dataset->output_size * sizeof(double));

        for (int j = 0; j < dataset->input_size; j++) {
            (*train_set)->inputs[i][j] = dataset->inputs[idx][j];
        }
        for (int j = 0; j < dataset->output_size; j++) {
            (*train_set)->targets[i][j] = dataset->targets[idx][j];
        }
    }

    // copy data to test set
    for (int i = 0; i < test_size; i++) {
        int idx = indices[train_size + i];
        (*test_set)->inputs[i] = (double*)malloc(dataset->input_size * sizeof(double));
        (*test_set)->targets[i] = (double*)malloc(dataset->output_size * sizeof(double));

        for (int j = 0; j < dataset->input_size; j++) {
            (*test_set)->inputs[i][j] = dataset->inputs[idx][j];
        }
        for (int j = 0; j < dataset->output_size; j++) {
            (*test_set)->targets[i][j] = dataset->targets[idx][j];
        }
    }

    free(indices);
}

// split dataset into train, validation, and test sets
void split_dataset_three_way(Dataset* dataset, double train_ratio, double val_ratio,
                             Dataset** train_set, Dataset** val_set, Dataset** test_set) {

    if (train_ratio < 0 || train_ratio > 1)
        train_ratio = 0.7;
    if (val_ratio < 0 || val_ratio > 1)
        val_ratio = 0.15;
    if (train_ratio + val_ratio > 1.0) {
        train_ratio = 0.7;
        val_ratio = 0.15;
    }

    int train_size = (int)(dataset->num_samples * train_ratio);
    int val_size = (int)(dataset->num_samples * val_ratio);
    int test_size = dataset->num_samples - train_size - val_size;

    // create sets
    *train_set = (Dataset*)malloc(sizeof(Dataset));
    (*train_set)->num_samples = train_size;
    (*train_set)->input_size = dataset->input_size;
    (*train_set)->output_size = dataset->output_size;
    (*train_set)->inputs = (double**)malloc(train_size * sizeof(double*));
    (*train_set)->targets = (double**)malloc(train_size * sizeof(double*));

    *val_set = (Dataset*)malloc(sizeof(Dataset));
    (*val_set)->num_samples = val_size;
    (*val_set)->input_size = dataset->input_size;
    (*val_set)->output_size = dataset->output_size;
    (*val_set)->inputs = (double**)malloc(val_size * sizeof(double*));
    (*val_set)->targets = (double**)malloc(val_size * sizeof(double*));

    *test_set = (Dataset*)malloc(sizeof(Dataset));
    (*test_set)->num_samples = test_size;
    (*test_set)->input_size = dataset->input_size;
    (*test_set)->output_size = dataset->output_size;
    (*test_set)->inputs = (double**)malloc(test_size * sizeof(double*));
    (*test_set)->targets = (double**)malloc(test_size * sizeof(double*));

    // shuffle indices
    int* indices = (int*)malloc(dataset->num_samples * sizeof(int));
    for (int i = 0; i < dataset->num_samples; i++)
        indices[i] = i;

    // fisher-yates shuffle
    for (int i = dataset->num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // copy data to train set
    for (int i = 0; i < train_size; i++) {
        int idx = indices[i];
        (*train_set)->inputs[i] = (double*)malloc(dataset->input_size * sizeof(double));
        (*train_set)->targets[i] = (double*)malloc(dataset->output_size * sizeof(double));

        for (int j = 0; j < dataset->input_size; j++) {
            (*train_set)->inputs[i][j] = dataset->inputs[idx][j];
        }
        for (int j = 0; j < dataset->output_size; j++) {
            (*train_set)->targets[i][j] = dataset->targets[idx][j];
        }
    }

    // copy data to validation set
    for (int i = 0; i < val_size; i++) {
        int idx = indices[train_size + i];
        (*val_set)->inputs[i] = (double*)malloc(dataset->input_size * sizeof(double));
        (*val_set)->targets[i] = (double*)malloc(dataset->output_size * sizeof(double));

        for (int j = 0; j < dataset->input_size; j++) {
            (*val_set)->inputs[i][j] = dataset->inputs[idx][j];
        }
        for (int j = 0; j < dataset->output_size; j++) {
            (*val_set)->targets[i][j] = dataset->targets[idx][j];
        }
    }

    // copy data to test set
    for (int i = 0; i < test_size; i++) {
        int idx = indices[train_size + val_size + i];
        (*test_set)->inputs[i] = (double*)malloc(dataset->input_size * sizeof(double));
        (*test_set)->targets[i] = (double*)malloc(dataset->output_size * sizeof(double));

        for (int j = 0; j < dataset->input_size; j++) {
            (*test_set)->inputs[i][j] = dataset->inputs[idx][j];
        }
        for (int j = 0; j < dataset->output_size; j++) {
            (*test_set)->targets[i][j] = dataset->targets[idx][j];
        }
    }

    free(indices);
}

// shuffle dataset
void shuffle_dataset(Dataset* dataset) {
    for (int i = dataset->num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        // swap inputs
        double* temp_input = dataset->inputs[i];
        dataset->inputs[i] = dataset->inputs[j];
        dataset->inputs[j] = temp_input;

        // swap targets
        double* temp_target = dataset->targets[i];
        dataset->targets[i] = dataset->targets[j];
        dataset->targets[j] = temp_target;
    }
}

// free dataset memory
void free_dataset(Dataset* dataset) {
    if (!dataset)
        return;

    for (int i = 0; i < dataset->num_samples; i++) {
        free(dataset->inputs[i]);
        free(dataset->targets[i]);
    }
    free(dataset->inputs);
    free(dataset->targets);
    free(dataset);
}
