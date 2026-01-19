#ifndef DATASET_H
#define DATASET_H

typedef struct {
    double** inputs;
    double** targets;
    int num_samples;
    int input_size;
    int output_size;
} Dataset;

// dataset generation functions
Dataset* create_sine_wave_dataset(int num_samples);
Dataset* create_xor_dataset();
Dataset* create_circle_dataset(int num_samples);
Dataset* create_circle_dataset_enhanced(int num_samples, double boundary_ratio);

// utility functions
void split_dataset(Dataset* dataset, double train_ratio,
                   Dataset** train_set, Dataset** test_set);
void split_dataset_three_way(Dataset* dataset, double train_ratio, double val_ratio,
                             Dataset** train_set, Dataset** val_set, Dataset** test_set);
void shuffle_dataset(Dataset* dataset);
void free_dataset(Dataset* dataset);

#endif
