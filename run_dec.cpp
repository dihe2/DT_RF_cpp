#include <iostream>
#include "DecisionTree.h"

using std::cout;

int main(int argc, char* argv[]){
	
	if(argc != 3){
		cout << "Error parsing arguments.\n";
		return -1;
	}

	DecisionTree dt;
	if(!dt.parse_train_file(argv[1])){
		return -1;
	}
	dt.train();
	
	//test use
	//dt.visial();

	if(!dt.parse_test_file(argv[2])){
		return -1;
	}

	dt.print_conf_mat();
	return 0;
}
