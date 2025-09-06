#include "io_util.h"
#include <fstream>
#include <sstream>

std::vector<float> io_util::import_vector_csv(const std::string& path) {
    std::vector<float> vec;
    std::string line, cell;
    std::fstream input_file;

    input_file.open(path, std::ios::in);

	std::string temp;

	while (std::getline(input_file, line)) {

		while (!line.empty() && line.back() == ',') {
			line.pop_back();
		}

		// Skip empty lines
		if (line.empty()) {
			continue;
		}

		// Convert the line to float and add to the column vector
		vec.push_back(std::stod(line));
	}
	input_file.close();


	return vec;
}

std::vector<std::vector<float>> io_util::import_matrix_csv(const std::string& path){
    std::vector<std::vector<float>> mat;
	std::string line;
	std::fstream input_file;

	input_file.open(path, std::ios::in);

	std::string temp;

	while (std::getline(input_file, line)) {
		//std::cout << "Line: " << line << std::endl;
		std::istringstream linestream(line);
		std::string cell;
		std::vector<float> row;

		
		while (std::getline(linestream, cell, ',')) {
			if (!cell.empty() && cell.size() > 1) {
				//std::cout << cell << std::endl;
				row.push_back(std::stod(cell));
			}
		}
		//std::cout << row.size() << std::endl;
		if (row.size() > 0) {
			mat.push_back(row);
		}
	}
	input_file.close();


	return mat;
}