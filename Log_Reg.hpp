#ifndef _LOG_REG_H
#define _LOG_REG_H
#include<iostream>
#include<vector>

class LogisticRegression
{
public:
    std::vector<double> coeffs; //beta value
    double learning_rate;
    int max_iter;

    std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>>& z);
    std::tuple<std::vector<std::vector<double>>, double> initialization(const std::vector<std::vector<double>>& X);
    std::vector<std::vector<double>> model(const std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& W, double b);
    double log_loss(std::vector<std::vector<double>>& A, std::vector<int>& y);
    std::tuple<std::vector<std::vector<double>>, double> gradients(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& A, std::vector<int>& y);


};













#endif 