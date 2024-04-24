#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <string>
#include <chrono>
#include "functions.h"

using namespace Eigen;
using namespace std;


int main(int argc, char* argv[]) {
    int nn, p, nz_size;
    string fname;

    // validate parameters
    if (argc != 5)
    {
        cout << "Wrong number of inputs. Exiting program..." << endl;

        return -1;
    }else{
        // store input variables
        nn = stoi(argv[1]);
        p = stoi(argv[2]);
        nz_size = stoi(argv[3]);
        fname = argv[4];
    }

    // Create data
    auto data = create_data(nn, p, nz_size);
    MatrixXd X = data.first;
    VectorXd y = data.second;

    // --- ADMM ALGORITHM IMPLEMENTATION ---

    // declaring important parameters
    double lambda_max = (X * y).lpNorm<Infinity>();  // Calculate lambda_max
    double lambda_val = 0.1 * lambda_max;  // pre-requisite to the singular lasso problem
    double abs_tol = 1e-4;
    double rel_tol = 1e-2;
    double rho = 1.0;  // preselected and constant throughout iteration
    int max_iter = 25;  // pre-computation shows ADMM convergs at 14/25 iterations
    MatrixXd A = X.transpose(); // transform X to the right dimension
    VectorXd b = y; // copy y into the vector b

    int m = X.rows();
    int n = A.cols();

    // declaring update parameters
    VectorXd x = VectorXd::Zero(n), z = VectorXd::Zero(n), u = VectorXd::Zero(n);
    
    History history;
    //Implementing ADMM algorithm
    for (int k = 0; k < max_iter; ++k) {
        // start time of iteration
        auto start = chrono::high_resolution_clock::now();
        MatrixXd x_old = x;  // beta-update
        
        MatrixXd z_old = z;  // z-update
        
        // Update x
        x = (A.transpose() * A + rho * MatrixXd::Identity(n, n)).ldlt().solve(A.transpose() * b + rho * (z - u));

        // Update z
        z = soft_threshold(x + u, lambda_val / rho);

        // Update u
        u += x - z;

        // Save history
        history.beta.push_back(x);
        history.z.push_back(z);
        history.u.push_back(u);

        // Compute norms
        double r_norm = (x - z).norm();
        double s_norm = (-rho*(z - z_old)).norm();
        history.r_norm.push_back(r_norm);
        history.s_norm.push_back(s_norm);

        double eps_pri = std::sqrt(p) * abs_tol + rel_tol * std::max(x.norm(), z.norm());
        double eps_dual = std::sqrt(n) * abs_tol + rel_tol * u.norm();
        history.eps_pri.push_back(eps_pri);
        history.eps_dual.push_back(eps_dual);

        // end time and store in history         
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        history.time.push_back(elapsed.count()); // save elapsed time

        // print out the current iteration level
        // Output or use the history for analysis
        cout << "Iteration: #" << k << endl;
    }

    // save history data in csv
    saveHistoryToCSV(history, fname);

    // Output the completion of the ADMM algorithm
    cout << "ADMM finished." << endl;
    
    return 0;
}
