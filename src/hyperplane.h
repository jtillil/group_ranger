/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#ifndef HYPERPLANE_H_
#define HYPERPLANE_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <cstddef> 
#include <memory> 
#include <type_traits> 
#include <utility> 

#ifdef R_BUILD
#include <Rinternals.h>
#endif

#include "globals.h"
#include "Data.h"
#include "Eigen/Dense"

namespace ranger {

bool x_is_in_right_child(std::vector<double> x, std::vector<double> coefs, double val);

bool x_is_in_right_child_hyperplane(std::vector<double> x, std::vector<double> hyperplane);

// std::vector<double> 
bool LDA(Eigen::MatrixXd x1, Eigen::MatrixXd x2, std::vector<double>& hyperplane);

// boolean returns
// bool is_in_right_child;
// bool hyperplane_success;

}

#endif /* HYPERPLANE_H_ */