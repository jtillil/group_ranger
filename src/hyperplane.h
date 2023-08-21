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

namespace ranger {

std::vector<double> LDA(std::vector<double> x, std::vector<double> y);

}

#endif /* HYPERPLANE_H_ */