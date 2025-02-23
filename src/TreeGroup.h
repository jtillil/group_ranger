/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#ifndef TREEGROUP_H_
#define TREEGROUP_H_

// #include <Rcpp.h>
#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>

#include "globals.h"
#include "Data.h"

namespace ranger {

class TreeGroup {
public:
  TreeGroup();

  // Create from loaded forest
  TreeGroup(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_groupIDs,
      std::vector<double>& split_values, std::vector<std::vector<double>>& split_coefficients);

  virtual ~TreeGroup() = default;

  TreeGroup(const TreeGroup&) = delete;
  TreeGroup& operator=(const TreeGroup&) = delete;

  void init(const Data* data, uint mtry, size_t num_samples, uint seed, std::vector<size_t>* deterministic_varIDs,
      std::vector<double>* split_select_weights, ImportanceMode importance_mode, uint min_node_size, uint min_bucket,
      bool sample_with_replacement, bool memory_saving_splitting, SplitRule splitrule,
      std::vector<double>* case_weights, std::vector<size_t>* manual_inbag, bool keep_inbag,
      std::vector<double>* sample_fraction, double alpha, double minprop, bool holdout, uint num_random_splits,
      uint max_depth, std::vector<double>* regularization_factor, bool regularization_usedepth,
      std::vector<bool>* split_groupIDs_used, bool* use_grouped_variables,
    std::vector<std::vector<uint>>* groups, uint* num_groups, std::string* splitmethod, bool* debug);

  virtual void allocateMemory() = 0;

  void grow(std::vector<double>* variable_importance);

  void predict(const Data* prediction_data, bool oob_prediction);

  void computePermutationImportance(std::vector<double>& forest_importance, std::vector<double>& forest_variance,
      std::vector<double>& forest_importance_casewise);

  void appendToFile(std::ofstream& file);
  virtual void appendToFileInternal(std::ofstream& file) = 0;

  const std::vector<std::vector<size_t>>& getChildNodeIDs() const {
    return child_nodeIDs;
  }
  const std::vector<double>& getSplitValues() const {
    return split_values;
  }
  const std::vector<std::vector<double>>& getSplitCoefficients() const {
    return split_coefficients;
  }
  const std::vector<size_t>& getSplitGroupIDs() const {
    return split_groupIDs;
  }

  const std::vector<size_t>& getOobSampleIDs() const {
    return oob_sampleIDs;
  }
  size_t getNumSamplesOob() const {
    return num_samples_oob;
  }

  const std::vector<size_t>& getInbagCounts() const {
    return inbag_counts;
  }

protected:
  void createPossibleSplitGroupSubset(std::vector<size_t>& result);

  bool splitNode(size_t nodeID);
  virtual bool splitNodeInternal(size_t nodeID, std::vector<size_t> possible_split_groupIDs) = 0;

  void createEmptyNode();
  virtual void createEmptyNodeInternal() = 0;

  size_t dropDownSamplePermuted(size_t permuted_varID, size_t sampleID, size_t permuted_sampleID);
  void permuteAndPredictOobSamples(size_t permuted_varID, std::vector<size_t>& permutations);

  virtual double computePredictionAccuracyInternal(std::vector<double>* prediction_error_casewise) = 0;
  
  void bootstrap();
  // void bootstrapWithoutReplacement();

  // void bootstrapWeighted();
  // void bootstrapWithoutReplacementWeighted();

  virtual void bootstrapClassWise();
  virtual void bootstrapWithoutReplacementClassWise();

  void setManualInbag();

  virtual void cleanUpInternal() = 0;

  void regularize(double& decrease, size_t varID) {
    if (regularization) {
      if (importance_mode == IMP_GINI_CORRECTED) {
        varID = data->getUnpermutedVarID(varID);
      }
      if ((*regularization_factor)[varID] != 1) {
        if (!(*split_groupIDs_used)[varID]) {
          if (regularization_usedepth) {
            decrease *= std::pow((*regularization_factor)[varID], depth + 1);
          } else {
            decrease *= (*regularization_factor)[varID];
          }
        }
      }
    }
  }

  void regularizeNegative(double& decrease, size_t varID) {
      if (regularization) {
        if (importance_mode == IMP_GINI_CORRECTED) {
          varID = data->getUnpermutedVarID(varID);
        }
        if ((*regularization_factor)[varID] != 1) {
          if (!(*split_groupIDs_used)[varID]) {
            if (regularization_usedepth) {
              decrease /= std::pow((*regularization_factor)[varID], depth + 1);
            } else {
              decrease /= (*regularization_factor)[varID];
            }
          }
        }
      }
    }

  void saveSplitGroupID(size_t groupID) {
    if (regularization) {
      if (importance_mode == IMP_GINI_CORRECTED) {
        // Rcpp::Rcerr << "Error: " << "saveSplitGroupID not implemented for IMP_GINI_CORRECTED." << std::endl;
        (*split_groupIDs_used)[data->getUnpermutedVarID(groupID)] = true;
      } else {
        (*split_groupIDs_used)[groupID] = true;
      }
    }
  }

  uint mtry;

  // Number of samples (all samples, not only inbag for this tree)
  size_t num_samples;

  // Number of OOB samples
  size_t num_samples_oob;

  // Minimum node size to split, nodes of smaller size can be produced
  uint min_node_size;
  
  // Minimum bucket size, minimum number of samples in each node
  uint min_bucket;

  // Weight vector for selecting possible split variables, one weight between 0 (never select) and 1 (always select) for each variable
  // Deterministic variables are always selected
  const std::vector<size_t>* deterministic_varIDs;
  const std::vector<double>* split_select_weights;

  // Bootstrap weights
  const std::vector<double>* case_weights;

  // Pre-selected bootstrap samples
  const std::vector<size_t>* manual_inbag;

  // Splitting variable for each node
  std::vector<size_t> split_groupIDs;

  // Value to split at for each node, defines position of the separation hyperplane
  // For terminal nodes the prediction value is saved here
  std::vector<double> split_values;

  // Values to split at for each node, defines the separation hyperplane
  // For terminal nodes nothing is saved here
  std::vector<std::vector<double>> split_coefficients;

  // Vector of left and right child node IDs, 0 for no child
  std::vector<std::vector<size_t>> child_nodeIDs;

  // All sampleIDs in the tree, will be re-ordered while splitting
  std::vector<size_t> sampleIDs;

  // For each node a vector with start and end positions
  std::vector<size_t> start_pos;
  std::vector<size_t> end_pos;

  // IDs of OOB individuals, sorted
  std::vector<size_t> oob_sampleIDs;

  // Holdout mode
  bool holdout;

  // Inbag counts
  bool keep_inbag;
  std::vector<size_t> inbag_counts;

  // Random number generator
  std::mt19937_64 random_number_generator;

  // Pointer to original data
  const Data* data;

  // Regularization
  bool regularization;
  std::vector<double>* regularization_factor;
  bool regularization_usedepth;
  std::vector<bool>* split_groupIDs_used;
  
  // Variable importance for all variables
  std::vector<double>* variable_importance;
  ImportanceMode importance_mode;

  // When growing here the OOB set is used
  // Terminal nodeIDs for prediction samples
  std::vector<size_t> prediction_terminal_nodeIDs;

  bool sample_with_replacement;
  const std::vector<double>* sample_fraction;

  bool memory_saving_splitting;
  SplitRule splitrule;
  double alpha;
  double minprop;
  uint num_random_splits;
  uint max_depth;
  uint depth;
  size_t last_left_nodeID;

  // Group specific members
  bool* use_grouped_variables;
  const std::vector<std::vector<uint>>* groups;
  uint* num_groups;
  const std::string* splitmethod;
  bool* debug;
};

} // namespace ranger

#endif /* TREEGROUP_H_ */
