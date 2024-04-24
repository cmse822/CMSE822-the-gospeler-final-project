#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <string>
#include <chrono>

using namespace Eigen;
using namespace std;

// Function to generate data with flexibility 
pair<MatrixXd, VectorXd> create_data(int n=1500, int p=5000, int nz_size=100) {
    MatrixXd X(p, n);
    VectorXd beta_true(p);
    VectorXd y(n);

    // Random number generation
    default_random_engine generator;
    normal_distribution<double> distribution(0.0,1.0);
    normal_distribution<double> small_noise(0.0,0.001);

    // Generate X matrix
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < n; ++j) {
            X(i, j) = distribution(generator);
        }
    }

    // Normalize rows of X to have unit l2 norm
    X.colwise().normalize();

    // Setting up beta values
    beta_true.setZero();
    for (int i = 0; i < nz_size; ++i) {
        int idx = rand() % p;
        beta_true(idx) = distribution(generator);
    }

    // Generate noise and response vector y
    for (int i = 0; i < n; ++i) {
        y(i) = X.col(i).dot(beta_true) + small_noise(generator);
    }

    return {X, y};
}

// predefined data structure to store the iterated variables
struct History {
    vector<MatrixXd> beta;
    vector<MatrixXd> z;
    vector<MatrixXd> u;
    vector<double> r_norm;
    vector<double> s_norm;
    vector<double> eps_pri;
    vector<double> eps_dual;
    vector<double> time;
    
};

// function updates z
MatrixXd soft_threshold(const MatrixXd& x, double lambda) {
    return x.unaryExpr([&lambda](double y){ return std::max(0.0, abs(y) - lambda) * (y > 0 ? 1 : -1); });
}


// function to save the iterated variable in csv file
void saveHistoryToCSV(const History& history, const std::string& filename) {
    std::ofstream file(filename);
    
    // defining Headers
    file << "Beta,Z,U,R_Norm,S_Norm,Eps_Pri,Eps_Dual, Iteration_time\n";  // spaces removed for ease of analysis

    // Get the size of the longest vector to iterate through all elements
    size_t max_size = std::max({history.beta.size(), history.z.size(), history.u.size(), 
                                history.r_norm.size(), history.s_norm.size(), 
                                history.eps_pri.size(), history.eps_dual.size(), history.time.size()});

    for (size_t i = 0; i < max_size; ++i) {
        if (i < history.beta.size())
            file << history.beta[i](0, 0); 
        file << ",";

        if (i < history.z.size())
            file << history.z[i](0, 0); 
        file << ",";

        if (i < history.u.size())
            file << history.u[i](0, 0);
        file << ",";

        if (i < history.r_norm.size())
            file << history.r_norm[i];
        file << ",";

        if (i < history.s_norm.size())
            file << history.s_norm[i];
        file << ",";

        if (i < history.eps_pri.size())
            file << history.eps_pri[i];
        file << ",";

        if (i < history.eps_dual.size())
            file << history.eps_dual[i];
        file << ",";

        if (i < history.time.size())
            file << history.time[i];
        file << "\n";
    }

    file.close();
}