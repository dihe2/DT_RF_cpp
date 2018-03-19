#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <utility>
#include <set>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include "DecisionTree.h"

using std::map;
using std::vector;
using std::ifstream;
using std::stringstream;
using std::string;
using std::getline;
using std::cin;
using std::cout;
using std::pair;
using std::stoi;
using std::set;
using std::sort;
using std::set_difference;
using std::numeric_limits;
using std::endl;
using std::fill;
using std::rand;

//using namespace DecisionTree;

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

void DecisionTree::deleteTree(DT_node* curr_node){
	if(curr_node == NULL){
		return;
	}

	map<string, pair<DT_node*, int> >::iterator it = (curr_node->child_list).begin();

	if(it == (curr_node->child_list).end()){
		delete curr_node;
		return;
	}

	for(; it != (curr_node->child_list).end(); it++){
		deleteTree((it->second).first);
	}
}

int DecisionTree::recog_helper(DT_node* dt_node, train_data_point dp){
	if((dt_node->child_list).size() < 2){
		return (((dt_node->child_list).begin())->second).second;
	}

	map<string, pair<DT_node*, int> >::iterator it = (dt_node->child_list).find((dp.attr)[dt_node->att_ind]);
	if(it == (dt_node->child_list).end()){
		//if((dt_node->child_list).size() > 1){
			////cout << "Error: Value: " << (dp.attr)[dt_node->att_ind] << " for attr: " << dt_node->att_ind << " not found.\n";
			//return -1;
			//value not seen before, need to draw a random child
		unsigned int rand_child = rand()%(dt_node->child_list).size();
		//unsigned int rand_child = dt_node->defu_child;
		it = (dt_node->child_list).begin();
		for(unsigned int rand_counter = 0; rand_counter < rand_child; rand_counter++){
			it++;
		}
		//}else{	
		//	return (((dt_node->child_list).begin())->second).second;
		//}
		
	}
	if((it->second).first == NULL){
		return (it->second).second;
	}
	return recog_helper((it->second).first, dp);
}

bool DecisionTree::recog_data_point(train_data_point dp){
	int temp_output = recog_helper(root, dp) - 1;
	if(temp_output < 0 || temp_output >= (int) label_list.size()){
		//cout << "Invalid output label: " << temp_output + 1 << endl;
		return false;
	}
	conf_mat[dp.label - 1][temp_output]++;
	return true;
}

bool DecisionTree::parse_test_file(const char* fileName){
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
		//label_list.insert(temp_label);
		for(unsigned int attr_counter = 1; attr_counter < in_split.size(); attr_counter++){
			vector<string> attr_pair;
			str_split(in_split[attr_counter], ':', attr_pair);
			unsigned int temp_attr = (unsigned int) stoi(attr_pair[0]);
			td.attr[temp_attr] = attr_pair[1];
			//attr_list.insert(temp_attr);
		}
		if(!recog_data_point(td)){
			//cout << "Testing Datapoint: " << dp_counter << endl;
		}
		dp_counter++;
	}

	inputfs.close();
	return true;
}

bool DecisionTree::parse_train_file(const char* fileName){
	ifstream inputfs(fileName);

	if(!inputfs.is_open()){
		//cout << "Error opening training file.\n";
		return false;
	}

	string parse_in;
	//unsigned int debug_counter = 0;
	while(getline(inputfs, parse_in)){
		//debug_counter++;
		//cout << debug_counter << endl;
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

void DecisionTree::compile_invers(const unsigned int attr_ind, const vector<unsigned int> index_sublist, map<string, vector<unsigned int> > & invers_map){
	for(unsigned int sublist_counter = 0; sublist_counter < index_sublist.size(); sublist_counter++){
		map<unsigned int, string>::iterator attr_it = ((train_data[index_sublist[sublist_counter]]).attr).find(attr_ind);
		map<string, vector<unsigned int> >::iterator it;
		if(attr_it == ((train_data[index_sublist[sublist_counter]]).attr).end()){
			//cout << "Warning, attribute not found for training sample #" << index_sublist[sublist_counter] << endl;
			//continue;
			//assume sparse representation, value of this attribute will be set to "0"
			it = invers_map.find("0");
			
		}else{
			it = invers_map.find(attr_it->second);
		}
		if(it == invers_map.end()){
			vector<unsigned int> temp_vect;
			temp_vect.push_back(index_sublist[sublist_counter]);
			invers_map[attr_it->second] = temp_vect;
		}else{
			(it->second).push_back(index_sublist[sublist_counter]);
		}
		
	}
}

void DecisionTree::compile_invers_label(map<string, vector<unsigned int> > invers_map, map<string, set<int> > & label_list){
	for(map<string, vector<unsigned int> >::iterator it = invers_map.begin(); it != invers_map.end(); it++){
		set<int> temp_set;
		for(vector<unsigned int>::iterator vect_it = (it->second).begin(); vect_it != (it->second).end(); vect_it++){
			temp_set.insert((train_data[*vect_it]).label);
		}
		label_list[it->first] = temp_set;
	}
}

double DecisionTree::compute_gini(map<string, vector<unsigned int> > & invers_map, unsigned int full_set_size){

	double gini = 0.0f;			
	for(map<string, vector<unsigned int> >::iterator it = invers_map.begin(); it != invers_map.end(); it++){
		map<int, unsigned int> freq_counter;
		for(unsigned int index_counter = 0; index_counter < (it->second).size(); index_counter++){
			int temp_label = (train_data[(it->second)[index_counter]]).label;
			map<int, unsigned int>::iterator count_it = freq_counter.find(temp_label);
			if(count_it == freq_counter.end()){
				freq_counter[temp_label] = 1;
			}else{
				freq_counter[temp_label]++;
			}
		}
		double set_val = 1.0f;
		for(map<int, unsigned int>::iterator val_it = freq_counter.begin(); val_it != freq_counter.end(); val_it++){
			double temp_val = ((double) (val_it->second))/(it->second).size();
			set_val -= temp_val*temp_val;
		}

		gini += (((double) (it->second).size())/full_set_size)*set_val;
	}

	return gini;	
}

int DecisionTree::call_vote(vector<unsigned int> index_list){
	map<int, unsigned int> label_count;
	for(vector<unsigned int>::iterator it = index_list.begin(); it != index_list.end(); it++){
		map<int, unsigned int>::iterator label_it = label_count.find((train_data[*it]).label);
		if(label_it == label_count.end()){
			label_count[(train_data[*it]).label] = 1;
		}else{
			(label_it->second)++;
		}
	}

	unsigned int max_count = 0;
	int sele_label;
	for(map<int, unsigned int>::iterator label_it = label_count.begin(); label_it != label_count.end(); label_it++){
		if((label_it->second) > max_count){
			sele_label = label_it->first;
			max_count = label_it->second;	
		}
	}
	
	//debug only
	////cout << sele_label << endl;		
	return sele_label;
}

void DecisionTree::train_helper(vector<unsigned int> remain_index, vector<unsigned int> remain_attr, DT_node* & curr_node){
	//remaining data count is 0, should not happen
	if(remain_index.size() == 0){
		//cout << "Training error.\n";
		return;
	}

	//remaining attr count is 0 but remaining data count is not, need to call for voting, should be handle by caller
	/*if(remain_attr.size() == 0){
		//cout << "Training error.\n";
		return;
	}*/

	//build sub_list
	if(remain_attr.size() == 0 || (remain_attr.size() == 1 && remain_index.size() < train_data.size()/MIN_SAMPLE_DIV)){
	//if(remain_attr.size() == 0){
		//unsigned int att_sele = *(remain_attr.begin());
		unsigned int att_sele = 0;
		curr_node = new DT_node;
		curr_node->att_ind = att_sele;
		curr_node->defu_child = 0;
		//call for voting
		pair<DT_node*, int> temp_pair;
		temp_pair.first = NULL;
		temp_pair.second = call_vote(remain_index);
		(curr_node->child_list)[""] = temp_pair;
		return;
	}


	unsigned int att_sele_ind;
	map<string, vector<unsigned int> > min_gini_invers_map;
	if(remain_attr.size() > 1){
		double min_gini = numeric_limits<double>::max();
		//double min_gini = 0.0001f;
		for(unsigned int attr_counter = 0; attr_counter < remain_attr.size(); attr_counter++){
			map<string, vector<unsigned int> > temp_invers_map;
			compile_invers(remain_attr[attr_counter], remain_index, temp_invers_map);
			double temp_gini = compute_gini(temp_invers_map, remain_index.size());
			//debug use only
			////cout << remain_attr[attr_counter] << ": " << temp_gini << endl;
			if(temp_gini < min_gini || (temp_gini == min_gini && rand()%2 == 0)){
				min_gini_invers_map = temp_invers_map;
				att_sele_ind = attr_counter;
				min_gini = temp_gini;
			}
		}
		//debug only
		////cout << endl;
	}else{
		att_sele_ind = 0;
		compile_invers(remain_attr[att_sele_ind], remain_index, min_gini_invers_map);
	}
			
	unsigned int att_sele = remain_attr[att_sele_ind];
	curr_node = new DT_node;
	curr_node->att_ind = att_sele;

	unsigned int temp_defu = 0;
	unsigned int val_size = 0;
	unsigned int defu_counter = 0;
	for(map<string, vector<unsigned int> >::iterator val_it = min_gini_invers_map.begin(); val_it != min_gini_invers_map.end(); val_it++){
		if((val_it->second).size() > val_size){
			temp_defu = defu_counter;
			val_size = (val_it->second).size();
		}
	}
	curr_node->defu_child = temp_defu;

	//map<string, vector<unsigned int> > invers_map;
	//compile_invers(att_sele, remain_index, invers_map);
	map<string, set<int> > label_list;
	compile_invers_label(min_gini_invers_map, label_list);


	remain_attr.erase(remain_attr.begin() + att_sele_ind);
	for(map<string, set<int> >::iterator label_it = label_list.begin(); label_it != label_list.end(); label_it++){
				
		pair<DT_node*, int> temp_pair;
		temp_pair.first = NULL;
		temp_pair.second = 1;
		if((label_it->second).size() <= 1){
			temp_pair.second = *((label_it->second).begin());
			//(curr_node->child_list)[label_it->first] = temp_pair;
			//continue;
		}else{
			
			train_helper(min_gini_invers_map[label_it->first], remain_attr, temp_pair.first);
		}
		(curr_node->child_list)[label_it->first] = temp_pair;
	}
}

void DecisionTree::train(){
	vector<unsigned int> remain_index;
	for(unsigned int counter = 0; counter < train_data.size(); counter++){
		remain_index.push_back(counter);
	}
	
	vector<unsigned int> remain_attr(attr_list.begin(), attr_list.end());
			
	train_helper(remain_index, remain_attr, root);

			
}

void DecisionTree::print_tree(DT_node* dt_node, unsigned int curr_level){
	cout << "current level: " << curr_level << " attribute: " << dt_node->att_ind << " num of child: " << (dt_node->child_list).size() << endl;
	int child_counter = 1;
	for(map<string, pair<DT_node*, int> >::iterator it = (dt_node->child_list).begin(); it != (dt_node->child_list).end(); it++){
		if((*it).first.compare("") == 0){
			cout << "no child, label: " << (*it).second.second << endl;
			continue;
		}
		cout << "child #" << child_counter << " value: " << (*it).first;
		if((*it).second.first == NULL){
			cout << " label: " << (*it).second.second << endl;
		}else{
			cout << endl;
		}
		child_counter++;
	}

	curr_level++;
	for(map<string, pair<DT_node*, int> >::iterator it = (dt_node->child_list).begin(); it != (dt_node->child_list).end(); it++){
		if((*it).second.first != NULL){
			print_tree((*it).second.first, curr_level);
		}
	}
			
}

//debug use function
void DecisionTree::visial(){
	unsigned int level = 0;
	print_tree(root, level);
}	

void DecisionTree::print_conf_mat(){
	//cout << label_list.size() << endl;
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

DecisionTree::DecisionTree(){
	root = NULL;
	conf_mat = NULL;
	//debug_counter = 0;
}

DecisionTree::~DecisionTree(){
	deleteTree(root);
}
