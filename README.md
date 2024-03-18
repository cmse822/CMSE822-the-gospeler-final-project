[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14308448&assignment_repo_type=AssignmentRepo)
# CMSE 822: Parallel Computing Final Project

This will serve as the repo for your final project for the course. Complete all code development work in this repo. Commit early and often! For a detailed description of the project requirements, see [here](https://cmse822.github.io/projects).

First, give a brief description of your project topic idea and detail how you will address each of the specific project requierements given in the link above. 

### Alternating Direction Method of Multilpliers (ADMM) Algortithm for LASSO.
#### Project Description:

The least absolute shrinkage and selection operator (lasso) method is a popular method in statistics and machine learning for regression analysis with regularization. However, the lasso problem is not an easy solve due to its $L_1$ lasso penalty, unlike the linear regression or ridge regression algortithm. But, algorithms such as the ADMM, have been developed to estimte the lasso estimator. 
    $$\hat{\beta}_{\text{lasso}} = \arg\min_{\beta} \left\{ \frac{1}{2}\|y - X^T\beta\|^2 + \lambda\|\beta\|_1 \right\}.$$

Given a feature matrix $X^T \in \mathbb{R}^{n\times p}$, where $n = 1500$ examples and $p = 5000$ features.  
The data are generated such that $X_{i, j} \sim N(0, 1)$, with its rows normalized to have unit $l_2$ norm; A *true* value $\beta^{true} \in \mathbb{R}^p$, with 100 nonzero entries each sampled from an $N(0, 1)$ distribution; and the $y$ label is computed as $y =X\beta^{true} + \epsilon$, where $\epsilon \sim N(0, 10^{-3})$.  

**The goal of this project is parallelization strategies in the ADMM algorithm to solve a single lasso problem (i.e., $\lambda = 1$ value) and regularization path (i.e., $\lambda = 100$ values).**
