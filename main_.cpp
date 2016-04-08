#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include "Snap.h"
#include "linkdata.h"

using namespace std;

vector<string> LoadFilePath()
{
	// 定义路径
	string featurePath = "F:/DataSet/AMiner/2005-2010/second_part_train_05_09.txt";
	string labelPath = "F:/DataSet/AMiner/2005-2010/second_part_full.txt";
	string affPath = "F:/DataSet/AMiner/2005-2010/aff.csv";
	string tagPath = "F:/DataSet/AMiner/2005-2010/tag.csv";

	// 把路径放到一个vector
	vector<string> paths;
	paths.push_back(featurePath);
	paths.push_back(labelPath);
	paths.push_back(affPath);
	paths.push_back(tagPath);

	return paths;
}
string LoadOutPath()
{
	string path = "second_dataset_10-4hop.txt";
	return path;
}

int main()
{
	vector<string> paths = LoadFilePath();
	string stopWords = "of Of University university univ.";

	int hop = 2;
	linkdata myData(paths, stopWords, hop);

	myData.GenerateKCore(10);

	myData.ExtractFeatures();
	//myData.ExtractAffTagSim();

	string outPath = LoadOutPath();
	myData.OutputData(outPath);
	

	system("pause");
	return 0;
}