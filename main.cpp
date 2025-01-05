#include<iostream>
#include"Log_Reg.hpp"
#include<math.h>
#include <vector>
#include <random>
#include <tuple>



int main()
{

    LogisticRegression fun(10);
    
    std::vector<std::vector<double>> X = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    std::vector<int> y(3);
    std::cout<< "size of y "<< y.size() << std::endl;


    auto result = fun.initialization(X);

    std::vector<std::vector<double>> W = std::get<0>(result);
    double b = std::get<1>(result);

    // Print weights and bias
    std::cout << "Weights (W):\n"<< y.size();
    for (const auto& row : W) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Bias (b): " << b << std::endl;

    size_t rows = W.size();              
    size_t cols = (rows > 0) ? W[0].size() : 0; 
    std::cout << "Shape of W: (" << rows << ", " << cols << ")" << std::endl;


    //-----------------------------------------------------------------------------------//


    auto A = fun.model(X,W,b);

    std::cout << "Sigmoide (A):\n";
    for (const auto& row : A) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    size_t Arows = A.size();              
    size_t Acols = (Arows > 0) ? A[0].size() : 0; 
    std::cout << "Shape of A: (" << Arows << ", " << Acols << ")" << std::endl;

    //------------------------------------------------------------------------------------//


    auto loss = fun.log_loss(A, y);
    std::cout<< " loss " << loss << std::endl;

    //------------------------------------------------------------------------------------//

    auto result2 = fun.gradients(X,A,y);

    std::vector<std::vector<double>> dW = std::get<0>(result2);
    double db = std::get<1>(result2);

    for (const auto& row : dW){
        for (const auto& elem : row){
            std::cout<< elem << " "; 
        }
        std::cout << std::endl;
    }

    std::cout << "Bias gradient (db): " << db << std::endl;

//-------------------------------------------------------------------------------------------//

    for(int i=0; i<=10;i++){
        auto result3 = fun.update(dW,db,W,b);
        std::vector<std::vector<double>> W = std::get<0>(result2);
        double b = std::get<1>(result2);
        for (const auto& row : dW){
            for (const auto& elem : row){
                std::cout<< elem << " "; 
            }
        std::cout << std::endl;
        }

        std::cout << "Bias gradient (b): " << db << std::endl;
    }

 
}

