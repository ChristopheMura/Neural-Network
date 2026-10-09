#ifndef MYLIB_H
#define MYLIB_H

#include <iostream>
#include <vector>
#include <random>


constexpr int N_OUTPUT = 1; // sortie binaire : loss, dZ4 et seuillage ne gèrent que 1

struct NetworkParams
{
    std::vector<std::vector<double>> W1;
    std::vector<double> b1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b2;
    std::vector<std::vector<double>> W3;
    std::vector<double> b3;
    std::vector<std::vector<double>> W4;
    std::vector<double> b4;
};

struct Activations
{
    std::vector<std::vector<double>> A1;    // (n1, m)
    std::vector<std::vector<double>> A2;    // (n2, m)
    std::vector<std::vector<double>> A3;    // (n3, m)
    std::vector<std::vector<double>> A4;    // (n4, m)
    std::vector<std::vector<double>> Z1;    // (n1, m)
    std::vector<std::vector<double>> Z2;    // (n2, m)
    std::vector<std::vector<double>> Z3;    // (n3, m)
    std::vector<std::vector<double>> Z4;    // (n4, m)
};

struct Gradients
{
    std::vector<std::vector<double>> dW1;
    std::vector<double> db1;
    std::vector<std::vector<double>> dW2;
    std::vector<double> db2;
    std::vector<std::vector<double>> dW3;
    std::vector<double> db3;
    std::vector<std::vector<double>> dW4;
    std::vector<double> db4;
};

void Run(int choix, std::vector<std::vector<double>>& X, std::vector<int>& y, int N);

NetworkParams Initialisation(int n0, int n1, int n2, int n3, int n4);

Activations ForwardPropagation(std::vector<std::vector<double>>& X, NetworkParams& params);

double log_loss(std::vector<std::vector<double>>& A, std::vector<int> y);

Gradients BackPropagation(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params, Activations& activations);

void Update(NetworkParams& params, Gradients& gradients, double learning_rate);

NetworkParams Neural_network(std::vector<std::vector<double>>& X, std::vector<int>& y, int n1, int n2, int n3, int n4, double learning_rate=0.1, int n_iter=100);

std::vector<int> Predict(std::vector<std::vector<double>>& X, NetworkParams& params);

double Accuracy(std::vector<int>& y, std::vector<int>& y_pred);

void PlotDecisionBoundary(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params);

#endif