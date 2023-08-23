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

std::vector<double> LDA(Eigen::MatrixXf x, Eigen::VectorXf y) {

    // ## Calculate class means
    // mean0 <- colMeans(as.matrix(data_values[response == 0,]))
    // mean1 <- colMeans(as.matrix(data_values[response == 1,]))
    
    // ## Calculate coefficients and value
    // coefficients <- spdinv(mat) %*% (mean1 - mean0)
    // value <- sum(coefficients * (0.5*(mean1 + mean0)))

    // MatrixXd centered = mat.rowwise() - mat.colwise().mean();
    // MatrixXd cov = (centered.adjoint() * centered) / double(mat.rows() - 1);

    // Split up x matrix
    mat1 = 2;
    mat2 = 3;

    // Calculate class-specific means
    mean1 = ;
    mean2 = ;

    // Calculate class-specific covariance matrices
    Eigen::MatrixXf centered1 = mat1.rowwise() - mat1.colwise().mean();
    Eigen::MatrixXf covmat1 = (centered1.adjoint() * centered1) / double(mat1.rows() - 1);
    Eigen::MatrixXf centered2 = mat2.rowwise() - mat2.colwise().mean();
    Eigen::MatrixXf covmat2 = (centered2.adjoint() * centered2) / double(mat2.rows() - 1);

    // Weighted mean covariance matrix


    // Calculate coefs and val
    std::vector<double> hyperplane = {1};
    double val = (hyperplane * (0.5*(mean1 + mean0))).sum();
    hyperplane.push_back(val)

    return hyperplane;
}

}