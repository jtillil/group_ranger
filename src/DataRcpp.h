/*-------------------------------------------------------------------------------
 This file is part of Ranger.
 
 Ranger is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Ranger is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Ranger. If not, see <http://www.gnu.org/licenses/>.

Written by:

Marvin N. Wright
Institut für Medizinische Biometrie und Statistik
Universität zu Lübeck
Ratzeburger Allee 160
23562 Lübeck

http://www.imbs-luebeck.de
#-------------------------------------------------------------------------------*/

#ifndef DATARCPP_H_
#define DATARCPP_H_

#include <Rcpp.h>

#include "globals.h"
#include "utility.h"
#include "Data.h"

namespace ranger {

class DataRcpp: public Data {
public:
  DataRcpp() = default;
  DataRcpp(Rcpp::NumericMatrix& x, Rcpp::NumericMatrix& y, std::vector<std::string> variable_names, size_t num_rows, size_t num_cols) {
      this->x = x;
      this->y = y;
      this->variable_names = variable_names;
      this->num_rows = num_rows;
      this->num_cols = num_cols;
      this->num_cols_no_snp = num_cols;
    }
  
  DataRcpp(const DataRcpp&) = delete;
  DataRcpp& operator=(const DataRcpp&) = delete;
  
  virtual ~DataRcpp() override = default;
  
  double get_x(size_t row, size_t col) const override {
    // Use permuted data for corrected impurity importance
    size_t col_permuted = col;
    if (col >= num_cols) {
      col = getUnpermutedVarID(col);
      row = getPermutedSampleID(row);
    }
    
    if (col < num_cols_no_snp) {
      return x(row, col);
    } else {
      return getSnp(row, col, col_permuted);
    }
  }
  
  double get_y(size_t row, size_t col) const override {
    return y(row, col);
  }

  // WRITTEN FOR GROUP RANGER IMPLEMENTATION
  // std::vector<std::vector<double>> get_x_subset(std::vector<size_t> row, std::vector<uint> col) const override {

  //   printf("Started get_x_subset() in DataRcpp\n");

  //   std::vector<std::vector<double>> x_out;
  //   for (size_t i = 0; i < row.size(); ++i) {
  //     for (size_t j = 0; j < col.size(); ++j) {
  //       x_out[i].push_back(get_x(i, j));
  //     }
  //   }

  //   printf("Finished get_x_subset() in DataRcpp\n");

  //   return x_out;

  // }

  std::vector<std::vector<double>> get_x_subset(std::vector<size_t> row, std::vector<uint> col, bool* debug) const override {

    if (*debug) {
    printf("Starting get_x_subset() in DataRcpp\n");
    }

    std::vector<std::vector<double>> x_out(row.size(), std::vector<double>(col.size()));
    for (size_t i = 0; i < row.size(); ++i) {
      for (size_t j = 0; j < col.size(); ++j) {
        x_out[i][j] = get_x(row[i], col[j]);
      }
    }

    if (*debug) {
    printf("%zu\n", x_out.size());
    printf("%zu\n", x_out[0].size());
    printf("Finished get_x_subset() in DataRcpp\n");
    }

    return x_out;

  }

  // Get sample with columns specified by col, usually a group when use_variable_groups == TRUE
  std::vector<double> get_x_sample(size_t row, std::vector<uint> col, bool* debug) const override {

    if (*debug) {
    printf("Starting get_x_sample() in DataRcpp\n");
    }

    std::vector<double> x_out(std::vector<double>(col.size()));
    for (size_t j = 0; j < col.size(); ++j) {
      x_out[j] = get_x(row, col[j]);
    }

    if (*debug) {
    printf("%zu\n", x_out.size());
    printf("Finished get_x_sample() in DataRcpp\n");
    }

    return x_out;

  }

  std::vector<double> get_y_subset(std::vector<size_t> row) const override {

    std::vector<double> y_out;
    for (size_t i = 0; i < num_rows; ++i) {
      y_out.push_back(get_y(i, 0));
    }
    return y_out;

  }

  // #nocov start 
  void reserveMemory(size_t y_cols) override {
    // Not needed
  }
  
  void set_x(size_t col, size_t row, double value, bool& error) override {
    x(row, col) = value;
  }
  
  void set_y(size_t col, size_t row, double value, bool& error) override {
    y(row, col) = value;
  }
  // #nocov end 
  
private:
  Rcpp::NumericMatrix x;
  Rcpp::NumericMatrix y;
};

} // namespace ranger

#endif /* DATARCPP_H_ */
