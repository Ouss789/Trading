#include"Log_Reg.hpp"
#include<math.h>
#include <vector>
#include <random>
#include <tuple>



LogisticRegression::LogisticRegression(double learning_rate)
    : learning_rate(learning_rate) {}


std::vector<std::vector<double>> LogisticRegression::sigmoid(std::vector<std::vector<double>>& z) 
{
    std::vector<std::vector<double>> result(z.size(), std::vector<double>(z[0].size()));

    for (size_t i = 0; i < z.size(); ++i) { // Iterate through rows
        for (size_t j = 0; j < z[i].size(); ++j) { // Iterate through columns
            result[i][j] = 1.0 / (1.0 + std::exp(-z[i][j])); // Apply sigmoid element-wise
        }
    }

    return result; // Return the resulting vector
}

std::tuple<std::vector<std::vector<double>>, double> LogisticRegression::initialization(const std::vector<std::vector<double>>& X) 
{
    size_t n_features = X[0].size(); 

    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0); 

    std::vector<std::vector<double>> W(n_features, std::vector<double>(1));
    for (size_t i = 0; i < n_features; ++i) {
        W[i][0] = dist(gen);
    }

    double b = dist(gen);

    return std::make_tuple(W, b);
}

std::vector<std::vector<double>> LogisticRegression::model(const std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& W, double b)
{
    if (X[0].size() != W.size()) {
        std::cout<< X[0].size() << W.size() << std::endl;
        throw std::invalid_argument("Les dimensions des matrices ne permettent pas un produit matriciel.");
    }

    std::vector<std::vector<double>> z (X.size(), std::vector<double>(W[0].size()));
    for (size_t i = 0; i < X.size(); ++i) {           // Pour chaque ligne de X
        for (size_t j = 0; j < W[0].size(); ++j) {       // Pour chaque colonne de W
            for (size_t k = 0; k < X[0].size(); ++k) {   // Pour chaque élément à multiplier
                z[i][j] += X[i][k] * W[k][j] + b;
            }
        }
    }

    std::vector<std::vector<double>> A = sigmoid(z);

    return A;
}


double LogisticRegression::log_loss(std::vector<std::vector<double>>& A, std::vector<int>& y)
{
    double loss = 0.0;
    int len = y.size();

    for (int i = 0; i < len; ++i) {
        loss -= y[i] * std::log(A[i][0]) + (1 - y[i]) * std::log(1 - A[i][0]);
    }

    return loss / len;
}

std::tuple<std::vector<std::vector<double>>, double> LogisticRegression::gradients(
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& A,
    std::vector<int>& y)
{
   
    int rows = X.size();
    int cols = X[0].size();

    std::vector<std::vector<double>> X_t(cols, std::vector<double>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            X_t[j][i] = X[i][j];
        }
    }

    std::vector<std::vector<double>> dW(cols, std::vector<double>(1, 0.0));
    std::vector<std::vector<double>> diff(rows, std::vector<double>(1, 0.0));
    double db = 0.0;

    int len = y.size();

    
    for (int i = 0; i < rows; ++i) {
        diff[i][0] = A[i][0] - y[i];
        db += diff[i][0];
    }
    db /= len; 

    
    for (size_t i = 0; i < cols; ++i) {
        for (size_t k = 0; k < rows; ++k) {
            dW[i][0] += X_t[i][k] * diff[k][0];
        }
        dW[i][0] /= len; 
    }

    return std::make_tuple(dW, db);
}

std::tuple<std::vector<std::vector<double>>, double> LogisticRegression::update(std::vector<std::vector<double>>& dW, double& db, std::vector<std::vector<double>>& W, double& b)
{
    int rows = W.size();
    int cols = W[0].size();

    for (size_t i = 0; i < W.size(); ++i) {
        for (size_t j = 0; j < W[i].size(); ++j) {
            W[i][j] -= learning_rate * dW[i][j]; 
        }
    }
    b -= learning_rate * db; 

    return std::make_tuple(W,b);
}
