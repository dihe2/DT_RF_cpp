#include <iostream>
#include "RandomForest.h"

using std::cout;

int main(int argc, char* argv[]){
	
	if(argc != 3){
		cout << "Error parsing arguments.\n";
		return -1;
	}

	RandomForest rf;
	if(!rf.parse_train_file(argv[1])){
		return -1;
	}
	rf.train();

	if(!rf.parse_test_file(argv[2])){
		return -1;
	}

	rf.print_conf_mat();
	return 0;
}
