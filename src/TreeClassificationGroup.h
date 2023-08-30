/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#ifndef TREECLASSIFICATIONGROUP_H_
#define TREECLASSIFICATIONGROUP_H_

#include <vector>

#include "globals.h"
#include "TreeGroup.h"

namespace ranger {

class TreeClassificationGroup: public TreeGroup {
public:
  // TreeClassificationGroup(std::vector<double>* class_values, std::vector<uint>* response_classIDs,
  //     std::vector<std::vector<size_t>>* sampleIDs_per_class, std::vector<double>* class_weights);
  TreeClassificationGroup(std::vector<double>* class_values, std::vector<uint>* response_classIDs,
      std::vector<std::vector<size_t>>* sampleIDs_per_class, std::vector<double>* class_weights,
      bool* use_grouped_variables, std::vector<std::vector<uint>>* groups,
      uint* num_groups, std::string* splitmethod);

  // Create from loaded forest
  TreeClassificationGroup(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_groupIDs,
      std::vector<double>& split_values, std::vector<std::vector<double>>& split_coefficients,
      std::vector<double>* class_values, std::vector<uint>* response_classIDs);

  TreeClassificationGroup(const TreeClassificationGroup&) = delete;
  TreeClassificationGroup& operator=(const TreeClassificationGroup&) = delete;

  virtual ~TreeClassificationGroup() override = default;

  void allocateMemory() override;

  double estimate(size_t nodeID);
  void computePermutationImportanceInternal(std::vector<std::vector<size_t>>* permutations);
  void appendToFileInternal(std::ofstream& file) override;

  double getPrediction(size_t sampleID) const {
    size_t terminal_nodeID = prediction_terminal_nodeIDs[sampleID];
    return split_values[terminal_nodeID];
  }

  size_t getPredictionTerminalNodeID(size_t sampleID) const {
    return prediction_terminal_nodeIDs[sampleID];
  }

private:
  bool splitNodeInternal(size_t nodeID, std::vector<size_t> possible_split_groupIDs) override;
  void createEmptyNodeInternal() override;

  double computePredictionAccuracyInternal(std::vector<double>* prediction_error_casewise) override;

  // Called by splitNodeInternal(). Sets split_groupIDs and split_values.
  bool findBestSplit(size_t nodeID, std::vector<size_t> possible_split_groupIDs);
  void findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_classes,
      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
      double& best_decrease);
  void findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_classes,
      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
      double& best_decrease, const std::vector<double>& possible_split_values, std::vector<size_t>& counter_per_class,
      std::vector<size_t>& counter);
  void findBestSplitValueLargeQ(size_t nodeID, size_t varID, size_t num_classes,
      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
      double& best_decrease);
  void findBestSplitValueUnordered(size_t nodeID, size_t groupID, size_t num_classes,
      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, std::vector<double>& best_coefficients, 
      size_t& best_groupID, double& best_decrease);

  bool findBestSplitExtraTrees(size_t nodeID, std::vector<size_t>& possible_split_groupIDs);
  void findBestSplitValueExtraTrees(size_t nodeID, size_t varID, size_t num_classes,
      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
      double& best_decrease);
  void findBestSplitValueExtraTrees(size_t nodeID, size_t varID, size_t num_classes,
      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
      double& best_decrease, const std::vector<double>& possible_split_values, std::vector<size_t>& class_counts_right,
      std::vector<size_t>& n_right);
  void findBestSplitValueExtraTreesUnordered(size_t nodeID, size_t varID, size_t num_classes,
      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
      double& best_decrease);

  void addGiniImportance(size_t nodeID, size_t varID, double decrease);

  void bootstrapClassWise() override;
  void bootstrapWithoutReplacementClassWise() override;

  void cleanUpInternal() override {
    counter.clear();
    counter.shrink_to_fit();
    counter_per_class.clear();
    counter_per_class.shrink_to_fit();
  }

  // Classes of the dependent variable and classIDs for responses
  const std::vector<double>* class_values;
  const std::vector<uint>* response_classIDs;
  const std::vector<std::vector<size_t>>* sampleIDs_per_class;

  // Splitting weights
  const std::vector<double>* class_weights;

  // Group specific members
  bool* use_grouped_variables;
  const std::vector<std::vector<uint>>* groups;
  uint* num_groups;
  const std::string* splitmethod;

  std::vector<size_t> counter;
  std::vector<size_t> counter_per_class;
};

} // namespace ranger

#endif /* TreeClassificationGroup_H_ */
