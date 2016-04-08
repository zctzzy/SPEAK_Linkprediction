#pragma once
#include <vector>
#include <string>
#include <map>
#include "Snap.h"
using namespace std;

typedef vector<string> vs;
typedef vector<vs> vvs;

class linkdata
{
public:
	linkdata();
	linkdata(vs paths, string _stopWords, int _hop);
	void ExtractFeatures();
	void ExtractSP();
	void OutputData(string path);
	~linkdata();

	void GenerateKCore(int k);
	TIntH GetCore();
	PNEANet GetNet();
	PUNGraph GetFeatureSet();
	PUNGraph GetLabelSet();
	vector<int> GetSP();
	vector<int> GetLabel();
	vector<int> GetCC();
	void ExtractAffTagSim();
	vector<double> GetTag();
	vector<double> GetAff();
private:
	map<int, vs> ReadAttributes(string path);
	vs TextParse(string str);
	TIntFltH PageRankValue();
	double GenerateAA(PUNGraph g, TIntV nbrs);
	double GenerateKatz(PUNGraph g, int src, int dst);
	double GenerateSim(vs src, vs dst);
	vs GenerateVacabList(vs src, vs dst);
	vector<double> Word2Vec(vs src, vs vacabList);
	bool WordMeaning(string word);
	int NewEdge(int src, int dst);
	double GeneratePropflow(int src, int dst, int l);
	int GenerateSocialPattern(double src, double dst, double flag, TIntV nbrs);
	map<pair<int, int>, int> GenerateEdgeWeight();
	int GenerateCollaborationCloseness(int src, int dst, TIntV nbrs);
private:
	PNEANet net; // network with multiple parallel edges
	PUNGraph ugFeature; // undirected graph to obtain features
	PUNGraph ugLabel;
	string stopWords;
	double ugFlag;
	string affPath;
	string tagPath;
	int hop_m;
	//vvs rawAttrData;

	TIntH core; // core authors
	// features;
	vector<int> srcDegree_m;
	vector<int> dstDegree_m;
	vector<int> srcPapers_m;
	vector<int> dstPapers_m;
	vector<double> srcPRValue_m;
	vector<double> dstPRValue_m;
	vector<int> cn_m;
	vector<double> jc_m;
	vector<double> aa_m;
	vector<double> katz_m;
	vector<double> tagSim_m;
	vector<double> affSim_m;
	vector<double> propflow_m;
	vector<int> sp_m;
	vector<int> pa_m;
	vector<int> label_m;
	vector<int> realtion_m;

	map<int, vs> id_tag;
	map<int, vs> id_aff;
	map<pair<int, int>, int> weight;
	TIntFltH pagerank;
	TIntFltH pr;

};

