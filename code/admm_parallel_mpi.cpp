#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <string>
#include <chrono>
#include "functions.h"
#include <mpi.h>
#include <omp.h>

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

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create data
    auto data = create_data(nn, p, nz_size);
    MatrixXd X = data.first;
    VectorXd b = data.second;

    MatrixXd A = X.transpose();  // correcting the dimensions of A

    int m = A.rows();
    int n = A.cols();

    // shared variable
    int row_size = m/(size);  // rows per rank
    int rows_rem = m % (size);

    // add the remaining row fraction to the last thread aiding dynamic load balancing
    if (rank == size-1){
        row_size += rows_rem;
    }

    double A_local[row_size][m];  // where m is the total row size
    double b_local[row_size][m];

    MPI_Scatter(A.data(), m*row_size, MPI_DOUBLE, A_local, row_size * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b.data(), m*row_size, MPI_DOUBLE, A_local, row_size * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // declaring important parameters
    double lambda_max = (A.transpose()* b).lpNorm<Infinity>();  // Calculate lambda_max
    double lambda_val = 0.1 * lambda_max;  // pre-requisite to the singular lasso problem
    double abs_tol = 1e-4;
    double rel_tol = 1e-2;
    double rho = 1.0;  // preselected and constant throughout iteration
    int max_iter = 25;  // pre-computation shows ADMM convergs at 14/25 iterations
    double start, elapsed;


    // declaring update parameters
    VectorXd x = VectorXd::Zero(n), z = VectorXd::Zero(n), u = VectorXd::Zero(n);
    
    
    History history;
    //Implementing ADMM algorithm
    for (int k = 0; k < max_iter; ++k) {
        // start time of iteration
        if (rank == 0)
        {
            start = MPI_Wtime();
        }
             
        MatrixXd z_old = z;  // z-update
        
        // Update x
        x = (A.transpose() * A + rho * MatrixXd::Identity(n, n)).ldlt().solve(A.transpose() * b + rho * (z - u));

        // Update z
        z = soft_threshold(x + u, lambda_val / rho);

        // Update u
        u += x - z;
        // Compute norms
        double r_norm = (x - z).norm();
        double s_norm = (-rho*(z - z_old)).norm();
        
        double eps_pri = std::sqrt(p) * abs_tol + rel_tol * std::max(x.norm(), z.norm());
        double eps_dual = std::sqrt(n) * abs_tol + rel_tol * u.norm();

        // Save history
        
        history.beta.push_back(x);
        history.z.push_back(z);
        history.u.push_back(u);

        history.r_norm.push_back(r_norm);
        history.s_norm.push_back(s_norm);

        history.eps_pri.push_back(eps_pri);
        history.eps_dual.push_back(eps_dual);

        // end time and store in history 
        if (rank == 0)
        {
            elapsed = MPI_Wtime() - start;
            history.time.push_back(elapsed); // save elapsed time
            
        }
        

        

        // print out the current iteration level
        // Output or use the history for analysis
        if(rank == 0){
            cout << "Iteration: #" << k << endl;

        }
        
    }

    if (rank == 0)
    {
        
        // save history data in csv
        saveHistoryToCSV(history, fname);
    }
    
    


    // Output the completion of the ADMM algorithm
    if(rank == 0){
        cout << "ADMM finished." << endl;
    }
    MPI_Finalize();
    return 0;
}
