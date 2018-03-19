#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <vector>
#include <utility>
#include <string>
#include "DecisionTree.h"

using std::vector;
using std::set;
using std::string;

#define TRAIN_DIVISION 801
#define MIN_SAMPLE_DIV_RF 25

void str_split(const string input, const char delim, vector<string> & output);

class RandomForestTree: public DecisionTree{
	public:
		void load_train_data(vector<train_data_point> train_data);
		int recog_handler(train_data_point dp);
	protected:
		void train_helper(vector<unsigned int> remain_index, vector<unsigned int> remain_attr, DT_node* & curr_node);
		//int recog_helper(DT_node* dt_node, train_data_point dp);
};

class RandomForest{
	private:
		vector<RandomForestTree> forest;
		vector<train_data_point> train_data;
		set<unsigned int> attr_list;
		set<int> label_list;
		unsigned int tree_count;
		unsigned int** conf_mat;

		vector<train_data_point> draw_train_data();		
		bool recog_data_point(train_data_point dp);
	
	public: 
		bool parse_train_file(char* fileName);
		bool parse_test_file(char* fileName);
		void train();
		void print_conf_mat();
};

#endif
