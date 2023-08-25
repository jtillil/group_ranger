/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <math.h>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyperplane.h"
#include "utility.h"
#include "globals.h"
#include "Data.h"
#include "Eigen/Dense"

namespace ranger {

bool x_is_in_right_child(std::vector<double> x, std::vector<double> coefs, double val) {
    // Multiply x and coefs
    double hyperplaneval = 0;
    for (int i = 0; i < coefs.size(); ++i) {
        hyperplaneval += x[i]*coefs[i];
    }

    // Compare to val
    if (hyperplaneval > val) {
        is_in_right_child = true;
    } else {
        is_in_right_child = false;
    }

    return is_in_right_child;
}

bool x_is_in_right_child_hyperplane(std::vector<double> x, std::vector<double> hyperplane) {
    // Get val
    double val = hyperplane.back();
    hyperplane.pop_back();

    // Multiply x and coefs
    double hyperplaneval = 0;
    for (int i = 0; i < hyperplane.size(); ++i) {
        hyperplaneval += x[i]*hyperplane[i];
    }

    // Compare to val
    if (hyperplaneval > val) {
        bool is_in_right_child = true;
    } else {
        bool is_in_right_child = false;
    }

    return is_in_right_child;
}

// std::vector<double> 
bool LDA(Eigen::MatrixXf x1, Eigen::MatrixXf x2, std::vector<double>& hyperplane) {

    // ## Calculate class means
    // mean0 <- colMeans(as.matrix(data_values[response == 0,]))
    // mean1 <- colMeans(as.matrix(data_values[response == 1,]))
    
    // ## Calculate coefficients and value
    // coefficients <- spdinv(mat) %*% (mean1 - mean0)
    // value <- sum(coefficients * (0.5*(mean1 + mean0)))

    // MatrixXd centered = mat.rowwise() - mat.colwise().mean();
    // MatrixXd cov = (centered.adjoint() * centered) / double(mat.rows() - 1);

    // Calculate class-specific means and sizes
    Eigen::VectorXf mean1 = x1.colwise().mean();
    Eigen::VectorXf mean2 = x2.colwise().mean();

    // Calculate class-specific covariance matrices
    Eigen::MatrixXf centered1 = x1.rowwise() - mean1;
    Eigen::MatrixXf covmat1 = (centered1.adjoint() * centered1) / double(x1.rows() - 1);
    Eigen::MatrixXf centered2 = x2.rowwise() - mean2;
    Eigen::MatrixXf covmat2 = (centered2.adjoint() * centered2) / double(x2.rows() - 1);

    // Weighted mean covariance matrix
    Eigen::MatrixXf covmat = x1.rows()/(x1.rows()+x2.rows()) * covmat1 + x2.rows()/(x1.rows()+x2.rows()) * covmat2;

    // Calculate coefs and val
    Eigen::VectorXf coefs = covmat.inverse() * (mean1 - mean2);
    double val = (coefs * (0.5*(mean1 + mean2))).sum();

    // Convert Eigen vector to double
    // std::vector<double> hyperplane;
    hyperplane.reserve(coefs.size() + 1);  // Reserve space for efficiency
    for (int i = 0; i < coefs.size(); ++i) {
        hyperplane.push_back(static_cast<double>(coefs[i]));  // Convert and add to coefs
    }
    
    // Append val
    hyperplane.push_back(val);

    bool hyperplane_success = true;

    return hyperplane_success;
}

}