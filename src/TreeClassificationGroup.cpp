/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <Rcpp.h>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include "DataRcpp.h"
#include "TreeClassificationGroup.h"
#include "utility.h" 
#include "Data.h"
#include "hyperplane.h"

namespace ranger {

// TreeClassificationGroup::TreeClassificationGroup(std::vector<double>* class_values, std::vector<uint>* response_classIDs,
//     std::vector<std::vector<size_t>>* sampleIDs_per_class, std::vector<double>* class_weights) :
//     class_values(class_values), response_classIDs(response_classIDs), sampleIDs_per_class(sampleIDs_per_class), class_weights(
//         class_weights), counter(0), counter_per_class(0) {
TreeClassificationGroup::TreeClassificationGroup(std::vector<double>* class_values, std::vector<uint>* response_classIDs,
    std::vector<std::vector<size_t>>* sampleIDs_per_class, std::vector<double>* class_weights,
    bool* use_grouped_variables, std::vector<std::vector<uint>>* groups,
    uint* num_groups, std::string* splitmethod) :
    class_values(class_values), response_classIDs(response_classIDs), sampleIDs_per_class(sampleIDs_per_class), class_weights(
        class_weights), use_grouped_variables(use_grouped_variables), groups(groups), num_groups(num_groups), splitmethod(
        splitmethod), counter(0), counter_per_class(0) {
}

TreeClassificationGroup::TreeClassificationGroup(std::vector<std::vector<size_t>>& child_nodeIDs,
    std::vector<size_t>& split_groupIDs, std::vector<double>& split_values, std::vector<std::vector<double>>& split_coefficients, std::vector<double>* class_values,
    std::vector<uint>* response_classIDs) :
    TreeGroup(child_nodeIDs, split_groupIDs, split_values, split_coefficients), class_values(class_values), response_classIDs(response_classIDs), sampleIDs_per_class(
        0), class_weights(0), counter { }, counter_per_class { } {
}

void TreeClassificationGroup::allocateMemory() {
  // Init counters if not in memory efficient mode
  if (!memory_saving_splitting) {
    size_t num_classes = class_values->size();
    size_t max_num_splits = data->getMaxNumUniqueValues();

    // Use number of random splits for extratrees
    if (splitrule == EXTRATREES && num_random_splits > max_num_splits) {
      max_num_splits = num_random_splits;
    }

    counter.resize(max_num_splits);
    counter_per_class.resize(num_classes * max_num_splits);
  }
}

double TreeClassificationGroup::estimate(size_t nodeID) {

  // Count classes over samples in node and return class with maximum count
  std::vector<double> class_count = std::vector<double>(class_values->size(), 0.0);

  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    size_t value = (*response_classIDs)[sampleID];
    class_count[value] += (*class_weights)[value];
  }

  if (end_pos[nodeID] > start_pos[nodeID]) {
    size_t result_classID = mostFrequentClass(class_count, random_number_generator);
    return ((*class_values)[result_classID]);
  } else {
    throw std::runtime_error("Error: Empty node.");
  }

}

void TreeClassificationGroup::appendToFileInternal(std::ofstream& file) { // #nocov start
  // Empty on purpose
} // #nocov end

bool TreeClassificationGroup::splitNodeInternal(size_t nodeID, std::vector<size_t> possible_split_groupIDs) {

  // Stop if maximum node size or depth reached
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  if (num_samples_node <= min_node_size || (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth)) {
    split_values[nodeID] = estimate(nodeID);
    return true;
  }

  // Check if node is pure
  bool pure = true;
  double pure_value = 0;
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    double value = data->get_y(sampleID, 0);
    if (pos != start_pos[nodeID] && value != pure_value) {
      pure = false;
      break;
    }
    pure_value = value;
  }
  // Set split_value to estimate and stop if pure
  if (pure) {
    split_values[nodeID] = pure_value;
    return true;
  }

  // Find best split, stop if no decrease of impurity
  bool stop;
  if (splitrule == EXTRATREES) {
    stop = findBestSplitExtraTrees(nodeID, possible_split_groupIDs);
  } else {
    stop = findBestSplit(nodeID, possible_split_groupIDs);
  }

  if (stop) {
    split_values[nodeID] = estimate(nodeID);
    return true;
  }

  return false;
}

void TreeClassificationGroup::createEmptyNodeInternal() {
  // Empty on purpose
}

double TreeClassificationGroup::computePredictionAccuracyInternal(std::vector<double>* prediction_error_casewise) {

  size_t num_predictions = prediction_terminal_nodeIDs.size();
  size_t num_missclassifications = 0;
  for (size_t i = 0; i < num_predictions; ++i) {
    size_t terminal_nodeID = prediction_terminal_nodeIDs[i];
    double predicted_value = split_values[terminal_nodeID];
    double real_value = data->get_y(oob_sampleIDs[i], 0);
    if (predicted_value != real_value) {
      ++num_missclassifications;
      if (prediction_error_casewise) {
        (*prediction_error_casewise)[i] = 1;
      }
    } else {
      if (prediction_error_casewise) {
        (*prediction_error_casewise)[i] = 0;
      }
    }
  }
  return (1.0 - (double) num_missclassifications / (double) num_predictions);
}

bool TreeClassificationGroup::findBestSplit(size_t nodeID, std::vector<size_t> possible_split_groupIDs) {

  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  size_t num_classes = class_values->size();
  double best_decrease = -1;
  size_t best_groupID = 0;
  double best_value = 0;
  std::vector<double> best_coefficients = {};

  std::vector<size_t> class_counts(num_classes);
  // Compute overall class counts
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    uint sample_classID = (*response_classIDs)[sampleID];
    ++class_counts[sample_classID];
  }

// Stop early if no split posssible
  if (num_samples_node >= 2 * min_bucket) {

    // For all possible split groups
    for (size_t groupID : possible_split_groupIDs) {
      // Find best split value, if ordered consider all values as split values, else all 2-partitions
      // if (data->isOrderedVariable(varID)) {

      //   Use memory saving method if option set
      //   if (memory_saving_splitting) {
      //     findBestSplitValueSmallQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
      //         best_decrease);
      //   } else {
      //     // Use faster method for both cases
      //     double q = (double) num_samples_node / (double) data->getNumUniqueDataValues(varID);
      //     if (q < Q_THRESHOLD) {
      //       findBestSplitValueSmallQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
      //           best_decrease);
      //     } else {
      //       findBestSplitValueLargeQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
      //           best_decrease);
      //     }
      //   }
      // } else {
        findBestSplitValueUnordered(nodeID, groupID, num_classes, class_counts, num_samples_node, best_value, best_coefficients, best_groupID,
            best_decrease);
            //(&groups)[groupID], splitmethod);
      // }
    }
  }

  // Stop if no good split found
  if (best_decrease < 0) {
    return true;
  }

  // Save best values
  split_groupIDs[nodeID] = best_groupID;
  split_values[nodeID] = best_value;
  split_coefficients[nodeID] = best_coefficients;

  // Compute gini index for this node and to variable importance if needed
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    addGiniImportance(nodeID, best_groupID, best_decrease);
  }

  // Regularization
  saveSplitGroupID(best_groupID);

  return false;
}

void TreeClassificationGroup::findBestSplitValueUnordered(size_t nodeID, size_t groupID, size_t num_classes,
    const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, std::vector<double>& best_coefficients, size_t& best_groupID,
    double& best_decrease) {

  printf("Starting findBestSplitValueUnordered()\n");

  // Setup variables
  bool success = false;
  std::vector<std::vector<double>> x1;
  std::vector<std::vector<double>> x2;
  Eigen::MatrixXd x1Eigen;
  Eigen::MatrixXd x2Eigen;
  const std::vector<uint>& group = (*groups)[groupID];

  // Get group-specific x and node-specific y values
  if (*splitmethod == "LDA") {
    // Extract positions from y
    std::vector<size_t> current_sampleIDs(sampleIDs.begin() + start_pos[nodeID], sampleIDs.begin() + end_pos[nodeID] - 1);
    std::vector<double> y_vals = data->get_y_subset(current_sampleIDs);
    std::vector<size_t> sampleIDs1;
    std::vector<size_t> sampleIDs2;
    for (size_t i = 0; i < num_samples_node; ++i) {
      if (y_vals[i] == data->get_y(0,0)) {
        sampleIDs1.push_back(i);
      } else {
        sampleIDs2.push_back(i);
      }
    }
    
    // Map x to x1 and x2
    // std::vector<uint> local_group = {groups[groupID][1]};
    // // for (uint varID : groups[groupID]) {
    // for (uint i = 1; i < groups[groupID].size(); ++i) {
    //   local_group.push_back(groups[groupID][i]);
    // }
    x1 = data->get_x_subset(sampleIDs1, group);
    x2 = data->get_x_subset(sampleIDs2, group);

    // Convert to Eigen::MatrixXd
    // for (uint j = 0; j < std::max(x1[0].size(), x2[0].size()); ++j) {
    //     for (uint i = 0; i < x1.size(); ++i) {
    //         x1Eigen(i, j) = x1[i][j];
    //     }
    //     for (uint i = 0; i < x2.size(); ++i) {
    //         x2Eigen(i, j) = x2[i][j];
    //     }
    // }

    // Ensure that x1 and x2 are non-empty and have valid dimensions
    if (x1.empty() || x2.empty() || x1[0].empty() || x2[0].empty()) {
        Rcpp::Rcerr << "Error: x1 or x2 are empty or have invalid dimensions." << std::endl;
        return;
    }

    // Ensure proper initialization and size for x1Eigen and x2Eigen
    x1Eigen.resize(x1.size(), x1[0].size());
    x2Eigen.resize(x2.size(), x2[0].size());

    // Convert to Eigen::MatrixXd with bounds checking
    for (uint j = 0; j < std::max(x1[0].size(), x2[0].size()); ++j) {
        for (uint i = 0; i < x1.size(); ++i) {
            if (j < x1[i].size()) {
                x1Eigen(i, j) = x1[i][j];
            } else {
                Rcpp::Rcerr << "Error: Index out of bounds in x1 conversion to Eigen::MatrixXd." << std::endl;
                return;
            }
        }
        for (uint i = 0; i < x2.size(); ++i) {
            if (j < x2[i].size()) {
                x2Eigen(i, j) = x2[i][j];
            } else {
                Rcpp::Rcerr << "Error: Index out of bounds in x2 conversion to Eigen::MatrixXd." << std::endl;
                return;
            }
        }
    }
  }

  // printf(x1Eigen);
  // Rcpp::Rcout << x1Eigen << std::endl;
  // printf(x2Eigen);
  // Rcpp::Rcout << x2Eigen << std::endl;
  printf("%f", x1Eigen[0][0]);

  // Calculate split hyperplane
  std::vector<double> hyperplane;
  if (*splitmethod == "LDA") {
    bool success = LDA(x1Eigen, x2Eigen, hyperplane);
  } else {
    Rcpp::Rcerr << "Error: " << "unknown splitmethod for grouped variables." << " Ranger will EXIT now." << std::endl;
  }

  // Check if success
  if (!success) {
    return;
  }

  // Initialize
  std::vector<size_t> class_counts_right(num_classes);
  size_t n_right = 0;

  // Compute right class counts
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];

    if (x_is_in_right_child_hyperplane((data->get_x_subset({sampleID}, group))[0], hyperplane)) {
      uint sample_classID = (*response_classIDs)[sampleID];
      ++class_counts_right[sample_classID];
      ++n_right;
    }
  }
  size_t n_left = num_samples_node - n_right;

  // Stop if minimal bucket size reached
  if (n_left < min_bucket || n_right < min_bucket) {
    return;
  }

  // Calculate decrease
  double decrease;
  if (splitrule == HELLINGER) {
    // // TPR is number of outcome 1s in one node / total number of 1s
    // // FPR is number of outcome 0s in one node / total number of 0s
    // double tpr = (double) class_counts_right[1] / (double) class_counts[1];
    // double fpr = (double) class_counts_right[0] / (double) class_counts[0];

    // // Decrease of impurity
    // double a1 = sqrt(tpr) - sqrt(fpr);
    // double a2 = sqrt(1 - tpr) - sqrt(1 - fpr);
    // decrease = sqrt(a1 * a1 + a2 * a2);
    Rcpp::Rcerr << "Error: " << "hellinger splitrule not supported for grouped variables." << " Ranger will EXIT now." << std::endl;
    return;
  } else {
    // Sum of squares
    double sum_left = 0;
    double sum_right = 0;
    for (size_t j = 0; j < num_classes; ++j) {
      size_t class_count_right = class_counts_right[j];
      size_t class_count_left = class_counts[j] - class_count_right;

      sum_right += (*class_weights)[j] * class_count_right * class_count_right;
      sum_left += (*class_weights)[j] * class_count_left * class_count_left;
    }
    // Decrease of impurity
    decrease = sum_left / (double) n_left + sum_right / (double) n_right;
  }

  // Regularization
  // regularize(decrease, varID);

  // If better than before, use this
  if (decrease > best_decrease) {
    best_value = hyperplane.back();
    hyperplane.pop_back();
    best_coefficients = hyperplane;
    best_groupID = groupID;
    best_decrease = decrease;
  }
}

// void TreeClassificationGroup::findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_classes,
//     const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
//     double& best_decrease) {

//   // Create possible split values
//   std::vector<double> possible_split_values;
//   data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

//   // Try next variable if all equal for this
//   if (possible_split_values.size() < 2) {
//     return;
//   }

//   const size_t num_splits = possible_split_values.size();
//   if (memory_saving_splitting) {
//     std::vector<size_t> class_counts_right(num_splits * num_classes), n_right(num_splits);
//     findBestSplitValueSmallQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
//         best_decrease, possible_split_values, class_counts_right, n_right);
//   } else {
//     std::fill_n(counter_per_class.begin(), num_splits * num_classes, 0);
//     std::fill_n(counter.begin(), num_splits, 0);
//     findBestSplitValueSmallQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
//         best_decrease, possible_split_values, counter_per_class, counter);
//   }
// }

// void TreeClassificationGroup::findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_classes,
//     const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
//     double& best_decrease, const std::vector<double>& possible_split_values, std::vector<size_t>& counter_per_class,
//     std::vector<size_t>& counter) {

//   for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
//     size_t sampleID = sampleIDs[pos];
//     uint sample_classID = (*response_classIDs)[sampleID];
//     size_t idx = std::lower_bound(possible_split_values.begin(), possible_split_values.end(),
//         data->get_x(sampleID, varID)) - possible_split_values.begin();

//     ++counter_per_class[idx * num_classes + sample_classID];
//     ++counter[idx];
//   }

//   size_t n_left = 0;
//   std::vector<size_t> class_counts_left(num_classes);

//   // Compute decrease of impurity for each split
//   for (size_t i = 0; i < possible_split_values.size() - 1; ++i) {

//     // Stop if nothing here
//     if (counter[i] == 0) {
//       continue;
//     }

//     n_left += counter[i];

//     // Stop if right child empty
//     size_t n_right = num_samples_node - n_left;
//     if (n_right == 0) {
//       break;
//     }

//     // Stop if minimal bucket size reached
//     if (n_left < min_bucket || n_right < min_bucket) {
//       continue;
//     }

//     double decrease;
//     if (splitrule == HELLINGER) {
//       for (size_t j = 0; j < num_classes; ++j) {
//         class_counts_left[j] += counter_per_class[i * num_classes + j];
//       }

//       // TPR is number of outcome 1s in one node / total number of 1s
//       // FPR is number of outcome 0s in one node / total number of 0s
//       double tpr = (double) (class_counts[1] - class_counts_left[1]) / (double) class_counts[1];
//       double fpr = (double) (class_counts[0] - class_counts_left[0]) / (double) class_counts[0];

//       // Decrease of impurity
//       double a1 = sqrt(tpr) - sqrt(fpr);
//       double a2 = sqrt(1 - tpr) - sqrt(1 - fpr);
//       decrease = sqrt(a1 * a1 + a2 * a2);
//     } else {
//       // Sum of squares
//       double sum_left = 0;
//       double sum_right = 0;
//       for (size_t j = 0; j < num_classes; ++j) {
//         class_counts_left[j] += counter_per_class[i * num_classes + j];
//         size_t class_count_right = class_counts[j] - class_counts_left[j];

//         sum_left += (*class_weights)[j] * class_counts_left[j] * class_counts_left[j];
//         sum_right += (*class_weights)[j] * class_count_right * class_count_right;
//       }

//       // Decrease of impurity
//       decrease = sum_right / (double) n_right + sum_left / (double) n_left;
//     }

//     // Regularization
//     regularize(decrease, varID);

//     // If better than before, use this
//     if (decrease > best_decrease) {
//       // Use mid-point split
//       best_value = (possible_split_values[i] + possible_split_values[i + 1]) / 2;
//       best_varID = varID;
//       best_decrease = decrease;

//       // Use smaller value if average is numerically the same as the larger value
//       if (best_value == possible_split_values[i + 1]) {
//         best_value = possible_split_values[i];
//       }
//     }
//   }
// }

// void TreeClassificationGroup::findBestSplitValueLargeQ(size_t nodeID, size_t varID, size_t num_classes,
//     const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
//     double& best_decrease) {

//   // Set counters to 0
//   size_t num_unique = data->getNumUniqueDataValues(varID);
//   std::fill_n(counter_per_class.begin(), num_unique * num_classes, 0);
//   std::fill_n(counter.begin(), num_unique, 0);

//   // Count values
//   for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
//     size_t sampleID = sampleIDs[pos];
//     size_t index = data->getIndex(sampleID, varID);
//     size_t classID = (*response_classIDs)[sampleID];

//     ++counter[index];
//     ++counter_per_class[index * num_classes + classID];
//   }

//   size_t n_left = 0;
//   std::vector<size_t> class_counts_left(num_classes);

//   // Compute decrease of impurity for each split
//   for (size_t i = 0; i < num_unique - 1; ++i) {

//     // Stop if nothing here
//     if (counter[i] == 0) {
//       continue;
//     }

//     n_left += counter[i];

//     // Stop if right child empty
//     size_t n_right = num_samples_node - n_left;
//     if (n_right == 0) {
//       break;
//     }

//     // Stop if minimal bucket size reached
//     if (n_left < min_bucket || n_right < min_bucket) {
//       continue;
//     }

//     double decrease;
//     if (splitrule == HELLINGER) {
//       for (size_t j = 0; j < num_classes; ++j) {
//         class_counts_left[j] += counter_per_class[i * num_classes + j];
//       }

//       // TPR is number of outcome 1s in one node / total number of 1s
//       // FPR is number of outcome 0s in one node / total number of 0s
//       double tpr = (double) (class_counts[1] - class_counts_left[1]) / (double) class_counts[1];
//       double fpr = (double) (class_counts[0] - class_counts_left[0]) / (double) class_counts[0];

//       // Decrease of impurity
//       double a1 = sqrt(tpr) - sqrt(fpr);
//       double a2 = sqrt(1 - tpr) - sqrt(1 - fpr);
//       decrease = sqrt(a1 * a1 + a2 * a2);
//     } else {
//       // Sum of squares
//       double sum_left = 0;
//       double sum_right = 0;
//       for (size_t j = 0; j < num_classes; ++j) {
//         class_counts_left[j] += counter_per_class[i * num_classes + j];
//         size_t class_count_right = class_counts[j] - class_counts_left[j];

//         sum_left += (*class_weights)[j] * class_counts_left[j] * class_counts_left[j];
//         sum_right += (*class_weights)[j] * class_count_right * class_count_right;
//       }

//       // Decrease of impurity
//       decrease = sum_right / (double) n_right + sum_left / (double) n_left;
//     }

//     // Regularization
//     regularize(decrease, varID);

//     // If better than before, use this
//     if (decrease > best_decrease) {
//       // Find next value in this node
//       size_t j = i + 1;
//       while (j < num_unique && counter[j] == 0) {
//         ++j;
//       }

//       // Use mid-point split
//       best_value = (data->getUniqueDataValue(varID, i) + data->getUniqueDataValue(varID, j)) / 2;
//       best_varID = varID;
//       best_decrease = decrease;

//       // Use smaller value if average is numerically the same as the larger value
//       if (best_value == data->getUniqueDataValue(varID, j)) {
//         best_value = data->getUniqueDataValue(varID, i);
//       }
//     }
//   }
// }

bool TreeClassificationGroup::findBestSplitExtraTrees(size_t nodeID, std::vector<size_t>& possible_split_groupIDs) {

  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  size_t num_classes = class_values->size();
  double best_decrease = -1;
  size_t best_varID = 0;
  double best_value = 0;

  std::vector<size_t> class_counts(num_classes);
  // Compute overall class counts
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    uint sample_classID = (*response_classIDs)[sampleID];
    ++class_counts[sample_classID];
  }

  // Stop early if no split posssible
  if (num_samples_node >= 2 * min_bucket) {

    // For all possible split variables
    for (auto& varID : possible_split_groupIDs) {
      // Find best split value, if ordered consider all values as split values, else all 2-partitions
      if (data->isOrderedVariable(varID)) {
        findBestSplitValueExtraTrees(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
            best_decrease);
      } else {
        findBestSplitValueExtraTreesUnordered(nodeID, varID, num_classes, class_counts, num_samples_node, best_value,
            best_varID, best_decrease);
      }
    }
  }

  // Stop if no good split found
  if (best_decrease < 0) {
    return true;
  }

  // Save best values
  split_groupIDs[nodeID] = best_varID;
  split_values[nodeID] = best_value;

  // Compute gini index for this node and to variable importance if needed
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    addGiniImportance(nodeID, best_varID, best_decrease);
  }

  // Regularization
  saveSplitGroupID(best_varID);

  return false;
}

void TreeClassificationGroup::findBestSplitValueExtraTrees(size_t nodeID, size_t varID, size_t num_classes,
    const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
    double& best_decrease) {

  // Get min/max values of covariate in node
  double min;
  double max;
  data->getMinMaxValues(min, max, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Try next variable if all equal for this
  if (min == max) {
    return;
  }

  // Create possible split values: Draw randomly between min and max
  std::vector<double> possible_split_values;
  std::uniform_real_distribution<double> udist(min, max);
  possible_split_values.reserve(num_random_splits);
  for (size_t i = 0; i < num_random_splits; ++i) {
    possible_split_values.push_back(udist(random_number_generator));
  }
  if (num_random_splits > 1) {
    std::sort(possible_split_values.begin(), possible_split_values.end());
  }

  const size_t num_splits = possible_split_values.size();
  if (memory_saving_splitting) {
    std::vector<size_t> class_counts_right(num_splits * num_classes), n_right(num_splits);
    findBestSplitValueExtraTrees(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
        best_decrease, possible_split_values, class_counts_right, n_right);
  } else {
    std::fill_n(counter_per_class.begin(), num_splits * num_classes, 0);
    std::fill_n(counter.begin(), num_splits, 0);
    findBestSplitValueExtraTrees(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
        best_decrease, possible_split_values, counter_per_class, counter);
  }
}

void TreeClassificationGroup::findBestSplitValueExtraTrees(size_t nodeID, size_t varID, size_t num_classes,
    const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
    double& best_decrease, const std::vector<double>& possible_split_values, std::vector<size_t>& class_counts_right,
    std::vector<size_t>& n_right) {
  const size_t num_splits = possible_split_values.size();

  // Count samples in right child per class and possbile split
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    double value = data->get_x(sampleID, varID);
    uint sample_classID = (*response_classIDs)[sampleID];

    // Count samples until split_value reached
    for (size_t i = 0; i < num_splits; ++i) {
      if (value > possible_split_values[i]) {
        ++n_right[i];
        ++class_counts_right[i * num_classes + sample_classID];
      } else {
        break;
      }
    }
  }

  // Compute decrease of impurity for each possible split
  for (size_t i = 0; i < num_splits; ++i) {

    // Stop if one child empty
    size_t n_left = num_samples_node - n_right[i];
    if (n_left == 0 || n_right[i] == 0) {
      continue;
    }

    // Stop if minimal bucket size reached
    if (n_left < min_bucket || n_right[i] < min_bucket) {
      continue;
    }

    // Sum of squares
    double sum_left = 0;
    double sum_right = 0;
    for (size_t j = 0; j < num_classes; ++j) {
      size_t class_count_right = class_counts_right[i * num_classes + j];
      size_t class_count_left = class_counts[j] - class_count_right;

      sum_right += (*class_weights)[j] * class_count_right * class_count_right;
      sum_left += (*class_weights)[j] * class_count_left * class_count_left;
    }

    // Decrease of impurity
    double decrease = sum_left / (double) n_left + sum_right / (double) n_right[i];

    // Regularization
    regularize(decrease, varID);

    // If better than before, use this
    if (decrease > best_decrease) {
      best_value = possible_split_values[i];
      best_varID = varID;
      best_decrease = decrease;
    }
  }
}

void TreeClassificationGroup::findBestSplitValueExtraTreesUnordered(size_t nodeID, size_t varID, size_t num_classes,
    const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
    double& best_decrease) {

  size_t num_unique_values = data->getNumUniqueDataValues(varID);

  // Get all factor indices in node
  std::vector<bool> factor_in_node(num_unique_values, false);
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    size_t index = data->getIndex(sampleID, varID);
    factor_in_node[index] = true;
  }

  // Vector of indices in and out of node
  std::vector<size_t> indices_in_node;
  std::vector<size_t> indices_out_node;
  indices_in_node.reserve(num_unique_values);
  indices_out_node.reserve(num_unique_values);
  for (size_t i = 0; i < num_unique_values; ++i) {
    if (factor_in_node[i]) {
      indices_in_node.push_back(i);
    } else {
      indices_out_node.push_back(i);
    }
  }

  // Generate num_random_splits splits
  for (size_t i = 0; i < num_random_splits; ++i) {
    std::vector<size_t> split_subset;
    split_subset.reserve(num_unique_values);

    // Draw random subsets, sample all partitions with equal probability
    if (indices_in_node.size() > 1) {
      size_t num_partitions = (2ULL << (indices_in_node.size() - 1ULL)) - 2ULL; // 2^n-2 (don't allow full or empty)
      std::uniform_int_distribution<size_t> udist(1, num_partitions);
      size_t splitID_in_node = udist(random_number_generator);
      for (size_t j = 0; j < indices_in_node.size(); ++j) {
        if ((splitID_in_node & (1ULL << j)) > 0) {
          split_subset.push_back(indices_in_node[j]);
        }
      }
    }
    if (indices_out_node.size() > 1) {
      size_t num_partitions = (2ULL << (indices_out_node.size() - 1ULL)) - 1ULL; // 2^n-1 (allow full or empty)
      std::uniform_int_distribution<size_t> udist(0, num_partitions);
      size_t splitID_out_node = udist(random_number_generator);
      for (size_t j = 0; j < indices_out_node.size(); ++j) {
        if ((splitID_out_node & (1ULL << j)) > 0) {
          split_subset.push_back(indices_out_node[j]);
        }
      }
    }

    // Assign union of the two subsets to right child
    size_t splitID = 0;
    for (auto& idx : split_subset) {
      splitID |= 1ULL << idx;
    }

    // Initialize
    std::vector<size_t> class_counts_right(num_classes);
    size_t n_right = 0;

    // Count classes in left and right child
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      uint sample_classID = (*response_classIDs)[sampleID];
      double value = data->get_x(sampleID, varID);
      size_t factorID = floor(value) - 1;

      // If in right child, count
      // In right child, if bitwise splitID at position factorID is 1
      if ((splitID & (1ULL << factorID))) {
        ++n_right;
        ++class_counts_right[sample_classID];
      }
    }
    size_t n_left = num_samples_node - n_right;

    // Stop if minimal bucket size reached
    if (n_left < min_bucket || n_right < min_bucket) {
      continue;
    }

    // Sum of squares
    double sum_left = 0;
    double sum_right = 0;
    for (size_t j = 0; j < num_classes; ++j) {
      size_t class_count_right = class_counts_right[j];
      size_t class_count_left = class_counts[j] - class_count_right;

      sum_right += (*class_weights)[j] * class_count_right * class_count_right;
      sum_left += (*class_weights)[j] * class_count_left * class_count_left;
    }

    // Decrease of impurity
    double decrease = sum_left / (double) n_left + sum_right / (double) n_right;

    // Regularization
    regularize(decrease, varID);

    // If better than before, use this
    if (decrease > best_decrease) {
      best_value = splitID;
      best_varID = varID;
      best_decrease = decrease;
    }
  }
}

void TreeClassificationGroup::addGiniImportance(size_t nodeID, size_t varID, double decrease) {

  double best_decrease = decrease;
  if (splitrule != HELLINGER) {
    size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
    std::vector<size_t> class_counts;
    class_counts.resize(class_values->size(), 0);

    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      uint sample_classID = (*response_classIDs)[sampleID];
      class_counts[sample_classID]++;
    }
    double sum_node = 0;
    for (size_t i = 0; i < class_counts.size(); ++i) {
      sum_node += (*class_weights)[i] * class_counts[i] * class_counts[i];
    }

    double impurity_node = (sum_node / (double) num_samples_node);

    // Account for the regularization
    regularize(impurity_node, varID);

    best_decrease = decrease - impurity_node;
  }

  // No variable importance for no split variables
  size_t tempvarID = data->getUnpermutedVarID(varID);

  // Subtract if corrected importance and permuted variable, else add
  if (importance_mode == IMP_GINI_CORRECTED && varID >= data->getNumCols()) {
    (*variable_importance)[tempvarID] -= best_decrease;
  } else {
    (*variable_importance)[tempvarID] += best_decrease;
  }
}

void TreeClassificationGroup::bootstrapClassWise() {
  // Number of samples is sum of sample fraction * number of samples
  size_t num_samples_inbag = 0;
  double sum_sample_fraction = 0;
  for (auto& s : *sample_fraction) {
    num_samples_inbag += (size_t) num_samples * s;
    sum_sample_fraction += s;
  }

  // Reserve space, reserve a little more to be save)
  sampleIDs.reserve(num_samples_inbag);
  oob_sampleIDs.reserve(num_samples * (exp(-sum_sample_fraction) + 0.1));

  // Start with all samples OOB
  inbag_counts.resize(num_samples, 0);

  // Draw samples for each class
  for (size_t i = 0; i < sample_fraction->size(); ++i) {
    // Draw samples of class with replacement as inbag and mark as not OOB
    size_t num_samples_class = (*sampleIDs_per_class)[i].size();
    size_t num_samples_inbag_class = round(num_samples * (*sample_fraction)[i]);
    std::uniform_int_distribution<size_t> unif_dist(0, num_samples_class - 1);
    for (size_t s = 0; s < num_samples_inbag_class; ++s) {
      size_t draw = (*sampleIDs_per_class)[i][unif_dist(random_number_generator)];
      sampleIDs.push_back(draw);
      ++inbag_counts[draw];
    }
  }

  // Save OOB samples
  for (size_t s = 0; s < inbag_counts.size(); ++s) {
    if (inbag_counts[s] == 0) {
      oob_sampleIDs.push_back(s);
    }
  }
  num_samples_oob = oob_sampleIDs.size();

  if (!keep_inbag) {
    inbag_counts.clear();
    inbag_counts.shrink_to_fit();
  }
}

void TreeClassificationGroup::bootstrapWithoutReplacementClassWise() {
  // Draw samples for each class
  for (size_t i = 0; i < sample_fraction->size(); ++i) {
    size_t num_samples_class = (*sampleIDs_per_class)[i].size();
    size_t num_samples_inbag_class = round(num_samples * (*sample_fraction)[i]);

    shuffleAndSplitAppend(sampleIDs, oob_sampleIDs, num_samples_class, num_samples_inbag_class,
        (*sampleIDs_per_class)[i], random_number_generator);
  }
  num_samples_oob = oob_sampleIDs.size();

  if (keep_inbag) {
    // All observation are 0 or 1 times inbag
    inbag_counts.resize(num_samples, 1);
    for (size_t i = 0; i < oob_sampleIDs.size(); i++) {
      inbag_counts[oob_sampleIDs[i]] = 0;
    }
  }
}

} // namespace ranger
