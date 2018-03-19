#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <set>

#define MIN_SAMPLE_DIV 50

using std::vector;
using std::string;
using std::map;
using std::pair;
using std::set;

typedef struct DT_node DT_node;
typedef struct train_data_point train_data_point;

struct train_data_point{
	int label;
	map<unsigned int, string> attr;
};

struct DT_node{
	unsigned int att_ind;
	unsigned int defu_child;
	map<string, pair<DT_node*, int> > child_list;
};

inline void str_split(const string input, const char delim, vector<string> & output);

class DecisionTree{
	protected:
		DT_node* root;
		vector<train_data_point> train_data;
		set<unsigned int> attr_list;
		set<int> label_list;
		unsigned int** conf_mat;

		void deleteTree(DT_node* curr_node);
		int recog_helper(DT_node* dt_node, train_data_point dp);
		bool recog_data_point(train_data_point dp);
		void compile_invers(const unsigned int att_ind, const vector<unsigned int> index_sublist, map<string, vector<unsigned int> > & invers_map);
		void compile_invers_label(map<string, vector<unsigned int> > invers_map, map<string, set<int> > & label_list);
		double compute_gini(map<string, vector<unsigned int> > & invers_map, unsigned int full_set_size);
		int call_vote(vector<unsigned int> index_list);
		void train_helper(vector<unsigned int> remain_index, vector<unsigned int> remain_attr, DT_node* & curr_node);
		void print_tree(DT_node* dt_node, unsigned int curr_level);

	public:
		DecisionTree();
		~DecisionTree();
		bool parse_test_file(const char* fileName);
		bool parse_train_file(const char* fileName);
		void train();
		void visial();
		void print_conf_mat();
};
#endif
