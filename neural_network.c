#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// structure
typedef struct {
  int input_size;
  int hidden_size;
  int output_size;

  // weights and biases
  double **w1;
  double *b1;
  double **w2;
  double *b2;

  // activations
  double *hidden;
  double *output;

  // gradients
  double **dw1;
  double *db1;
  double **dw2;
  double *db2;
} NeuralNetwork;

// activation function
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// derivative of sigmoid
double sigmoid_derivative(double x) { return x * (1.0 - x); }

// initialize nn
NeuralNetwork *create_network(int input_size, int hidden_size,
                              int output_size) {
  NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
  nn->input_size = input_size;
  nn->hidden_size = hidden_size;
  nn->output_size = output_size;

  // random number generator
  srand(time(NULL));

  // allocate memory for weights and biases
  nn->w1 = (double **)malloc(hidden_size * sizeof(double *));
  nn->b1 = (double *)calloc(hidden_size, sizeof(double));
  nn->w2 = (double **)malloc(output_size * sizeof(double *));
  nn->b2 = (double *)calloc(output_size, sizeof(double));

  nn->dw1 = (double **)malloc(hidden_size * sizeof(double *));
  nn->db1 = (double *)calloc(hidden_size, sizeof(double));
  nn->dw2 = (double **)malloc(output_size * sizeof(double *));
  nn->db2 = (double *)calloc(output_size, sizeof(double));

  // allocate activations
  nn->hidden = (double *)calloc(hidden_size, sizeof(double));
  nn->output = (double *)calloc(output_size, sizeof(double));

  // initialize weights with random values
  for (int i = 0; i < hidden_size; i++) {
    nn->w1[i] = (double *)malloc(input_size * sizeof(double));
    nn->dw1[i] = (double *)calloc(input_size, sizeof(double));
    for (int j = 0; j < input_size; j++) {
      nn->w1[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
    }
  }

  for (int i = 0; i < output_size; i++) {
    nn->w2[i] = (double *)malloc(hidden_size * sizeof(double));
    nn->dw2[i] = (double *)calloc(hidden_size, sizeof(double));
    for (int j = 0; j < hidden_size; j++) {
      nn->w2[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
    }
  }

  return nn;
}

// forward propagation
void forward(NeuralNetwork *nn, double *input) {

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
void backward(NeuralNetwork *nn, double *input, double *target,
              double learning_rate) {

  // calculate output layer error and gradients
  double *output_error = (double *)malloc(nn->output_size * sizeof(double));
  for (int i = 0; i < nn->output_size; i++) {
    output_error[i] = target[i] - nn->output[i];

    double delta_output = output_error[i] * sigmoid_derivative(nn->output[i]);

    // update output layer weights
    for (int j = 0; j < nn->hidden_size; j++) {
      nn->dw2[i][j] = delta_output * nn->hidden[j];
      nn->w2[i][j] += learning_rate * nn->dw2[i][j];
    }
    nn->db2[i] = delta_output;
    nn->b2[i] += learning_rate * nn->db2[i];
  }

  // calculate hidden layer error and gradients
  for (int i = 0; i < nn->hidden_size; i++) {
    double hidden_error = 0;
    for (int j = 0; j < nn->output_size; j++) {
      hidden_error += output_error[j] * nn->w2[j][i];
    }

    double delta_hidden = hidden_error * sigmoid_derivative(nn->hidden[i]);

    // update hidden layer weights
    for (int j = 0; j < nn->input_size; j++) {
      nn->dw1[i][j] = delta_hidden * input[j];
      nn->w1[i][j] += learning_rate * nn->dw1[i][j];
    }
    nn->db1[i] = delta_hidden;
    nn->b1[i] += learning_rate * nn->db1[i];
  }

  free(output_error);
}

// calculate mean squared error
double calculate_mse(NeuralNetwork *nn, double **inputs, double **targets,
                     int num_samples) {
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
void free_network(NeuralNetwork *nn) {
  for (int i = 0; i < nn->hidden_size; i++) {
    free(nn->w1[i]);
    free(nn->dw1[i]);
  }
  free(nn->w1);
  free(nn->dw1);
  free(nn->b1);
  free(nn->db1);

  for (int i = 0; i < nn->output_size; i++) {
    free(nn->w2[i]);
    free(nn->dw2[i]);
  }
  free(nn->w2);
  free(nn->dw2);
  free(nn->b2);
  free(nn->db2);

  free(nn->hidden);
  free(nn->output);
  free(nn);
}

// training example: XOR problem
int main() {
  printf("Building Neural Network in C - XOR Problem\n");
  printf("===========================================\n\n");

  // training data
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  double targets[4][1] = {{0}, {1}, {1}, {0}};

  double *input_ptrs[4];
  double *target_ptrs[4];
  for (int i = 0; i < 4; i++) {
    input_ptrs[i] = inputs[i];
    target_ptrs[i] = targets[i];
  }

  // create network: 2 inputs, 4 hidden neurons, 1 output
  NeuralNetwork *nn = create_network(2, 4, 1);

  // training params
  int epochs = 10000;
  double learning_rate = 0.5;

  printf("Training network for %d epochs...\n", epochs);

  for (int epoch = 0; epoch < epochs; epoch++) {
    double total_error = 0;

    // train all 4 XOR samples
    for (int sample = 0; sample < 4; sample++) {
      forward(nn, inputs[sample]);
      backward(nn, inputs[sample], targets[sample], learning_rate);

      // calculate error
      double error = targets[sample][0] - nn->output[0];
      total_error += error * error;
    }

    // print progress every 1000 epochs
    if (epoch % 1000 == 0) {
      printf("Epoch %d: MSE = %.6f\n", epoch, total_error / 4);
    }
  }

  printf("\nFinal Results:\n");
  printf("--------------\n");

  // test the trained network
  for (int i = 0; i < 4; i++) {
    forward(nn, inputs[i]);
    printf("Input: [%.0f, %.0f] -> Output: %.4f (Expected: %.0f)\n",
           inputs[i][0], inputs[i][1], nn->output[0], targets[i][0]);
  }

  // calculate final MSE
  double final_mse = calculate_mse(nn, input_ptrs, target_ptrs, 4);
  printf("\nFinal MSE: %.6f\n", final_mse);

  // clean up
  free_network(nn);

  return 0;
}
