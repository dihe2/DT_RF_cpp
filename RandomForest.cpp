#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <sstream>
#include "RandomForest.h"

using std::vector;
using std::string;
using std::ifstream;
using std::rand;
using std::sqrt;
using std::cout;
using std::endl;
using std::fill;
using std::numeric_limits;
using std::rand;
using std::stringstream;
//using std::floor;

inline void str_split(const string input, const char delim, vector<string> & output){
        stringstream ss;
        ss.str(input);
        string temp;

        while(getline(ss, temp, delim)){
                if(temp.compare("") != 0){
                        output.push_back(temp);
                }
        }
}

bool RandomForest::parse_train_file(char* fileName){
	ifstream inputfs(fileName);

	if(!inputfs.is_open()){
		//cout << "Error opening training file.\n";
		return false;
	}

	string parse_in;
	while(getline(inputfs, parse_in)){
		if(parse_in.compare("") == 0){
			continue;
		}
		vector<string> in_split;
		str_split(parse_in, ' ', in_split);
		train_data_point td;
		int temp_label = stoi(in_split[0]);
		td.label = temp_label; 
		label_list.insert(temp_label);
		for(unsigned int attr_counter = 1; attr_counter < in_split.size(); attr_counter++){
			vector<string> attr_pair;
			str_split(in_split[attr_counter], ':', attr_pair);
			unsigned int temp_attr = (unsigned int) stoi(attr_pair[0]);
			td.attr[temp_attr] = attr_pair[1];
			attr_list.insert(temp_attr);
		}
		train_data.push_back(td);
	}

	inputfs.close();
	return true;
}

void RandomForestTree::load_train_data(vector<train_data_point> train_data_in){
	for(unsigned int train_data_counter = 0; train_data_counter < train_data_in.size(); train_data_counter++){
		label_list.insert(train_data_in[train_data_counter].label);
		for(map<unsigned int, string>::iterator it = train_data_in[train_data_counter].attr.begin(); it != train_data_in[train_data_counter].attr.end(); it++){
			attr_list.insert(it->first);
		}
		train_data.push_back(train_data_in[train_data_counter]);
	}
}

vector<train_data_point> RandomForest::draw_train_data(){
	vector<train_data_point> return_set;
	for(unsigned int set_count = 0; set_count < train_data.size(); set_count++){
		return_set.push_back(train_data[rand()%train_data.size()]);
	}
	return return_set;
}

/*void RandomForest::initalize_tree(){
	tree_count = attr_list.size()*(train_data.size()/TRAIN_DIVISION + 1);
	for(unsigned int tree_counter = 0; tree_counter < tree_count; tree_counter++){
		RandomForestTree tree;
		tree.load_train_data(draw_train_data());
		tree.train();
		forest.push_back(tree);
	}
}*/

void RandomForest::train(){
	tree_count = attr_list.size()*(train_data.size()/TRAIN_DIVISION + 1);
	for(unsigned int tree_counter = 0; tree_counter < tree_count; tree_counter++){
		RandomForestTree tree;
		tree.load_train_data(draw_train_data());
		tree.train();
		forest.push_back(tree);
	}
}

void RandomForestTree::train_helper(vector<unsigned int> remain_index, vector<unsigned int> remain_attr, DT_node* & curr_node){
	//remaining data count is 0, should not happen
	if(remain_index.size() == 0){
		//cout << "Training error.\n";
		return;
	}

	//build sub_list
	if(remain_attr.size() == 0 || (remain_attr.size() == 1 && rand()%2 == 0) || (remain_attr.size() == 1 && remain_index.size() < train_data.size()/MIN_SAMPLE_DIV_RF)){
		//unsigned int att_sele = *(remain_attr.begin());
		unsigned int att_sele = 0;
		curr_node = new DT_node;
		curr_node->att_ind = att_sele;
		//call for voting
		pair<DT_node*, int> temp_pair;
		temp_pair.first = NULL;
		temp_pair.second = call_vote(remain_index);
		(curr_node->child_list)[""] = temp_pair;
		return;
	}

	//sample a subset of attributes
	vector<unsigned int> left_out_attr;
	unsigned int attr_size = remain_attr.size();
	for(vector<unsigned int>::iterator attr_it = remain_attr.begin(); attr_it != remain_attr.end();){
		if(rand()%attr_size < sqrt((float) attr_size)){
			left_out_attr.push_back(*attr_it);
			vector<unsigned int>::iterator temp_it = attr_it;
			attr_it++;
			remain_attr.erase(temp_it);
		}
	}
	
	if(remain_attr.size() == 0){
		unsigned int rand_num = rand()%left_out_attr.size();
		remain_attr.push_back(left_out_attr[rand_num]);
		left_out_attr.erase(left_out_attr.begin() + rand_num);
	}

	unsigned int att_sele_ind;
	map<string, vector<unsigned int> > min_gini_invers_map;
	if(remain_attr.size() > 1){
		double min_gini = numeric_limits<double>::max();
		for(unsigned int attr_counter = 0; attr_counter < remain_attr.size(); attr_counter++){
			map<string, vector<unsigned int> > temp_invers_map;
			compile_invers(remain_attr[attr_counter], remain_index, temp_invers_map);
			double temp_gini = compute_gini(temp_invers_map, remain_index.size());
			if(temp_gini < min_gini){
				min_gini_invers_map = temp_invers_map;
				att_sele_ind = attr_counter;
				min_gini = temp_gini;
			}
		}
	}else{
		att_sele_ind = 0;
		compile_invers(remain_attr[att_sele_ind], remain_index, min_gini_invers_map);
	}
			
	unsigned int att_sele = remain_attr[att_sele_ind];
	curr_node = new DT_node;
	curr_node->att_ind = att_sele;

	map<string, set<int> > label_list;
	compile_invers_label(min_gini_invers_map, label_list);


	remain_attr.erase(remain_attr.begin() + att_sele_ind);
	for(unsigned int left_out_counter = 0; left_out_counter < left_out_attr.size(); left_out_counter++){
		remain_attr.push_back(left_out_attr[left_out_counter]);
	}

	for(map<string, set<int> >::iterator label_it = label_list.begin(); label_it != label_list.end(); label_it++){
				
		pair<DT_node*, int> temp_pair;
		temp_pair.first = NULL;
		temp_pair.second = 1;
		if((label_it->second).size() <= 1){
			temp_pair.second = *((label_it->second).begin());
			(curr_node->child_list)[label_it->first] = temp_pair;
			continue;
		}else{
			train_helper(min_gini_invers_map[label_it->first], remain_attr, temp_pair.first);
			(curr_node->child_list)[label_it->first] = temp_pair;
		}		
		//train_helper(min_gini_invers_map[label_it->first], remain_attr, ((curr_node->child_list)[label_it->first]).first);
	}
}

bool RandomForest::parse_test_file(char* fileName){
	ifstream inputfs(fileName);

	if(!inputfs.is_open()){
		//cout << "Error opening test file.\n";
		return false;
	}

	//initialize confusion matrix to 0 matrix
	conf_mat = new unsigned int*[label_list.size()];
	for(unsigned int label_counter = 0; label_counter < label_list.size(); label_counter++){
		conf_mat[label_counter] = new unsigned int[label_list.size()];
		fill(conf_mat[label_counter], conf_mat[label_counter] + label_list.size(), 0);
	}

	//debug use
	unsigned int dp_counter = 1; 
	
	string parse_in;
	while(getline(inputfs, parse_in)){
		if(parse_in.compare("") == 0){
			continue;
		}
		vector<string> in_split;
		str_split(parse_in, ' ', in_split);
		train_data_point td;
		int temp_label = stoi(in_split[0]);
		td.label = temp_label; 
		for(unsigned int attr_counter = 1; attr_counter < in_split.size(); attr_counter++){
			vector<string> attr_pair;
			str_split(in_split[attr_counter], ':', attr_pair);
			unsigned int temp_attr = (unsigned int) stoi(attr_pair[0]);
			td.attr[temp_attr] = attr_pair[1];
		}
		if(!recog_data_point(td)){
			//cout << "Train Datapoint: " << dp_counter << endl;
		}
		dp_counter++;
	}

	inputfs.close();
	return true;
}

int RandomForestTree::recog_handler(train_data_point dp){
	return recog_helper(root, dp) - 1;
}

bool RandomForest::recog_data_point(train_data_point dp){
	//collect output from all trees
	map<int, unsigned int> vote_pool;
	for(unsigned int tree_counter = 0; tree_counter < tree_count; tree_counter++){
		int temp_output = forest[tree_counter].recog_handler(dp);
		if(vote_pool.find(temp_output) == vote_pool.end()){
			vote_pool[temp_output] = 1;
		}else{
			vote_pool[temp_output]++;
		}
	}

	//vote for forest output
	unsigned int max_vote_count = 0;
	int max_cand = -1;
	for(map<int, unsigned int>::iterator vote_it = vote_pool.begin(); vote_it != vote_pool.end(); vote_it++){
		if(vote_it->first != -1 && (vote_it->second > max_vote_count || (vote_it->second == max_vote_count && rand()%2 == 0))){
			max_vote_count = vote_it->second;
			max_cand = vote_it->first;
		}
	}
	int temp_output = max_cand;

	if(temp_output < 0 || temp_output >= (int) label_list.size()){
		//cout << "Invalid output label: " << temp_output + 1 << endl;
		return false;
	}
	conf_mat[dp.label - 1][temp_output]++;
	return true;
}

void RandomForest::print_conf_mat(){
	for(unsigned int counter_x = 0; counter_x < label_list.size(); counter_x++){
		for(unsigned int counter_y = 0; counter_y < label_list.size(); counter_y++){
			cout << conf_mat[counter_x][counter_y] << " ";
		}
		cout << endl;
	}

	//delete conf_mat
	for(unsigned int label_counter = 0; label_counter < label_list.size(); label_counter++){
		delete[] conf_mat[label_counter];
	}
	delete[] conf_mat;
}
