all: DecisionTree RandomForest

DecisionTree: DecisionTree.o run_dec.o
	g++ -std=c++11 -g -Wall DecisionTree.o run_dec.o -o DecisionTree

RandomForest: RandomForest.o run_ran.o DecisionTree.o
	g++ -std=c++11 -g -Wall RandomForest.o run_ran.o DecisionTree.o -o RandomForest

RandomForest.o: RandomForest.h RandomForest.cpp DecisionTree.h
	g++ -std=c++11 -g -Wall -c RandomForest.cpp

run_ran.o: RandomForest.h run_ran.cpp
	g++ -std=c++11 -g -Wall -c run_ran.cpp

DecisionTree.o: DecisionTree.h DecisionTree.cpp
	g++ -std=c++11 -g -Wall -c DecisionTree.cpp

run_dec.o: DecisionTree.h run_dec.cpp
	g++ -std=c++11 -g -Wall -c run_dec.cpp

clean: 
	rm -f *.o DecisionTree
