#include "linkdata.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include "Snap.h"
using namespace std;

typedef vector<string> vs;
typedef vector<vs> vvs;

linkdata::linkdata()
{
}

linkdata::linkdata(vs paths, string _stopWords, int _hop)
{
	cout << "Loading data..." << endl;
	net = TSnap::LoadEdgeList<PNEANet>(paths[0].c_str(), 0, 1);
	TSnap::DelSelfEdges(net);

	weight = GenerateEdgeWeight();

	ugFeature = TSnap::ConvertGraph<PUNGraph>(net);
	TSnap::DelSelfEdges(ugFeature);

	ugLabel = TSnap::LoadEdgeList<PUNGraph>(paths[1].c_str(), 0, 1);
	TSnap::DelSelfEdges(ugLabel);

	stopWords = _stopWords;

	hop_m = _hop;

	affPath = paths[2];
	tagPath = paths[3];
}

map<pair<int, int>, int> linkdata::GenerateEdgeWeight()
{
	map<pair<int, int>, int> w;
	for (TNEANet::TEdgeI EI = net->BegEI(); EI < net->EndEI(); EI++)
	{
		int src = EI.GetSrcNId();
		int dst = EI.GetDstNId();

		pair<int, int> key = make_pair(src, dst);
		w[key]++;
	}
	return w;
}

map<int, vs> linkdata::ReadAttributes(string path)
{
	map<int, vs> attrData;
	ifstream in(path.c_str());
	string line;
	int i = 0;
	while (getline(in, line))
	{
		int key = core.GetKey(i);
		vs str = TextParse(line);
		attrData[key] = str;
		i++;
	}
	return attrData;
}

vs linkdata::TextParse(string str)
{
	size_t beg = 0;
	size_t incre = 0;
	vs text;
	while ((incre = str.find(' ', beg)) != string::npos)
	{
		string temp = str.substr(beg, incre - beg);
		text.push_back(temp);
		beg = incre + 1;
	}
	text.push_back(str.substr(beg, incre - beg));
	return text;
}

void linkdata::ExtractFeatures()
{
	// load attributes data
	id_aff = ReadAttributes(affPath);
	id_tag = ReadAttributes(tagPath);

	// pagerank value
	pagerank = PageRankValue();

	// search all the nodes in core and obtain the neighors of N hop
	int count = 0;
	for (TIntH::TIter it = core.BegI(); it < core.EndI(); it++)
	{
		int srcID = it->Key;
		int srcDeg = ugFeature->GetNI(srcID).GetDeg();
		int srcPub = net->GetNI(srcID).GetDeg();

		TIntV lenKNodes;
		//int hop = 2;
		TSnap::GetNodesAtHop(ugFeature, srcID, hop_m, lenKNodes);
		for (int i = 0; i < lenKNodes.Len(); i++)
		{
			bool isInCore = core.IsKey(lenKNodes[i]);
			bool isNotSearch = srcID < lenKNodes[i] ? 1 : 0;
			if (isInCore && isNotSearch)
			{
				count++;
				if (count % 1000 == 0)
				{
					cout << "Finished " << count << " samples" << endl;
				}
				srcDegree_m.push_back(srcDeg);
				srcPapers_m.push_back(srcPub);
				
				int dstID = lenKNodes[i];
				int dstDeg = ugFeature->GetNI(dstID).GetDeg();
				dstDegree_m.push_back(dstDeg);
				int dstPub = net->GetNI(dstID).GetDeg();
				dstPapers_m.push_back(dstPub);
				
				TIntV nbrs;
				if (hop_m == 2)
				{
					// common neighbors
					int cn = TSnap::GetCmnNbrs(ugFeature, srcID, dstID, nbrs);
					cn_m.push_back(cn);

					// jaccard's coefficient
					double jc = double(cn) / double(srcDeg + dstDeg - cn);
					jc_m.push_back(jc);

					// adamic/adar
					double aa = GenerateAA(ugFeature, nbrs);
					aa_m.push_back(aa);
				}
				
				// pa
				int pa = srcDeg*dstDeg;
				pa_m.push_back(pa);
				
				// katz
				double katz = GenerateKatz(ugFeature, srcID, dstID);
				katz_m.push_back(katz);
				
				// pagerank value
				double srcPR = pagerank.GetDat(srcID);
				srcPRValue_m.push_back(srcPR);
				double dstPR = pagerank.GetDat(dstID);
				dstPRValue_m.push_back(dstPR);
				
				// tag similarity
				double tagSim = GenerateSim(id_tag[srcID], id_tag[dstID]);
				tagSim_m.push_back(tagSim);
				
				// affiliation similarity
				double affSim = GenerateSim(id_aff[srcID], id_aff[dstID]);
				affSim_m.push_back(affSim);
				
				// propflow
				double propflow = GeneratePropflow(srcID, dstID, 5);
				propflow_m.push_back(propflow);
				
				// social pattern
				int sp = GenerateSocialPattern(srcPR, dstPR, ugFlag, nbrs);
				sp_m.push_back(sp);

				// author collaboration's closeness
				int cc = GenerateCollaborationCloseness(srcID, dstID, nbrs);
				realtion_m.push_back(cc);
				
				// label info.
				int label = NewEdge(srcID, dstID);
				label_m.push_back(label);
			}
		}
	}
}

void linkdata::ExtractAffTagSim()
{
	// load attributes data
	id_aff = ReadAttributes(affPath);
	id_tag = ReadAttributes(tagPath);

	// search all the nodes and obtain the neighbors of N hops
	int count = 0;
	for (TIntH::TIter it = core.BegI(); it < core.EndI(); it++)
	{
		int srcID = it->Key;
		TIntV lenKNodes;
		//int hop = 2;
		TSnap::GetNodesAtHop(ugFeature, srcID, hop_m, lenKNodes);
		for (int i = 0; i < lenKNodes.Len(); i++)
		{
			bool isInCore = core.IsKey(lenKNodes[i]);
			bool isNotSearch = srcID < lenKNodes[i] ? 1 : 0;
			if (isInCore && isNotSearch)
			{
				count++;
				if (count % 1000 == 0)
				{
					cout << "Finished " << count << " samples" << endl;
				}

				int dstID = lenKNodes[i];

				// tag similarity
				double tagSim = GenerateSim(id_tag[srcID], id_tag[dstID]);
				tagSim_m.push_back(tagSim);

				// affiliation similarity
				double affSim = GenerateSim(id_aff[srcID], id_aff[dstID]);
				affSim_m.push_back(affSim);
			}
		}
	}
}

int linkdata::GenerateCollaborationCloseness(int src, int dst, TIntV nbrs)
{
	int w1, w2;
	if (nbrs.Len() == 1)
	{
		int midID = nbrs[0].Val;
		w1 = weight[make_pair(src, midID)];
		w2 = weight[make_pair(dst, midID)];
	}
	else
	{
		w1 = 0;
		w2 = 0;
	}

	int relation;
	if (w1 == 1 && w2 == 1)
	{
		relation = 1;
	}
	else if (w1 == 1 && w2 > 1)
	{
		relation = 2;
	}
	else if (w1 > 1 && w2 == 1)
	{
		relation = 3;
	}
	else if (w1 > 1 && w2 > 1)
	{
		relation = 4;
	}
	else
	{
		relation = 5;
	}

	return relation;
}

int linkdata::GenerateSocialPattern(double src, double dst, double flag, TIntV nbrs)
{
	int s, m, t;
	if (nbrs.Len() == 1)
	{
		// do something
		int midID = nbrs[0].Val;
		double midPR = pr.GetDat(midID);
		s = src > flag ? 1 : 0;
		m = midPR > flag ? 1 : 0;
		t = dst > flag ? 1 : 0;
	}
	else
	{
		// do something
		s = src > flag ? 1 : 0;
		m = -1;
		t = dst > flag ? 1 : 0;
	}
	
	int pattern;
	if (m == 0 && s == 1 && t == 1)
	{
		pattern = 1;
	}
	else if (m == 1 && s == 1 && t == 1)
	{
		pattern = 2;
	}
	else if (m == 0 && s == 1 && t == 0)
	{
		pattern = 3;
	}
	else if (m == 0 && s == 0 && t == 1)
	{
		pattern = 4;
	}
	else if (m == 1 && s == 1 && t == 0)
	{
		pattern = 5;
	}
	else if (m == 1 && s == 0 && t == 1)
	{
		pattern = 6;
	}
	else if (m == 0 && s == 0 && t == 0)
	{
		pattern = 7;
	}
	else if (m == 1 && s == 0 && t == 0)
	{
		pattern = 8;
	}
	else if (m == -1 && s == 0 && t == 0)
	{
		pattern = 9;
	}
	else if (m == -1 && s == 0 && t == 1)
	{
		pattern = 10;
	}
	else if (m == -1 && s == 1 && t == 0)
	{
		pattern = 11;
	}
	else if (m == -1 && s == 1 && t == 1)
	{
		pattern = 12;
	}
	else
	{
		pattern = 13;
	}
	
	return pattern;
}

double linkdata::GeneratePropflow(int srcID, int dstID, int l)
{
	int nodesNum = ugFeature->GetMxNId();
	vector<double> scores(nodesNum);
	vector<bool> found(nodesNum);
	vector<int> search;

	found.at(srcID) = true;
	search.push_back(srcID);
	scores.at(srcID) = 1;

	for (int degree = 0; degree < l; ++degree)
	{
		vector<int> newSearch;
		for (vector<int>::const_iterator vIt = search.begin(); vIt != search.end(); ++vIt)
		{
			const int searchNode = *vIt;
			double sourceInput = scores.at(searchNode);
			TUNGraph::TNodeI NI = ugFeature->GetNI(searchNode);
			int outDeg = NI.GetDeg();
			double totalOutput = (double)outDeg;
			for (int i = 0; i < outDeg; i++)
			{
				const int nbr = NI.GetNbrNId(i);
				double probability = sourceInput*(1.0 / totalOutput);
				scores.at(nbr) += probability;
				if (!found.at(nbr))
				{
					found.at(nbr) = true;
					newSearch.push_back(nbr);
				}
			}
		}
		search.swap(newSearch);
	}
	return scores.at(dstID);
}

int linkdata::NewEdge(int src, int dst)
{
	bool isEdge = ugLabel->IsEdge(src, dst);
	bool isInTrain = ugFeature->IsEdge(src, dst);
	if (isEdge && !isInTrain)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

double linkdata::GenerateSim(vs src, vs dst)
{
	vs vacabList;
	vacabList = GenerateVacabList(src, dst);
	vector<double> srcV = Word2Vec(src, vacabList);
	vector<double> dstV = Word2Vec(dst, vacabList);

	double numerator = 0.0;
	double denominator1 = 0.0, denominator2 = 0.0;
	for (size_t i = 0; i < vacabList.size(); i++)
	{
		numerator += srcV[i] * dstV[i];
		denominator1 += srcV[i] * srcV[i];
		denominator2 += dstV[i] * dstV[i];
	}

	return numerator / (sqrt(denominator1 + 1)*sqrt(denominator2 + 1));
}

vs linkdata::GenerateVacabList(vs src, vs dst)
{
	set<string> vacabSet;
	for (size_t i = 0; i < src.size(); i++)
	{
		vacabSet.insert(src[i]);
	}

	for (size_t j = 0; j < dst.size(); j++)
	{
		vacabSet.insert(dst[j]);
	}

	vs vacabList;
	std::copy(vacabSet.begin(), vacabSet.end(), std::back_inserter(vacabList));
	return vacabList;
}

vector<double> linkdata::Word2Vec(vs src, vs vacabList)
{
	vector<double> mat(vacabList.size(), 0);

	for (size_t i = 0; i < src.size(); i++)
	{
		string word = src[i];
		bool isMeaningLess = WordMeaning(word);
		if (!isMeaningLess)
		{
			size_t idx = std::find(vacabList.begin(), vacabList.end(), word) - vacabList.begin();
			if (idx != string::npos)
			{
				mat.at(idx) += 1;
			}
		}
	}
	return mat;
}

bool linkdata::WordMeaning(string word)
{
	return (stopWords.find(word) != string::npos) ? true : false;
}

double linkdata::GenerateKatz(PUNGraph g, int src, int dst)
{
	double beta = 0.05;
	int len2 = TSnap::GetLen2Paths(g, src, dst);
	TUNGraph::TNodeI srcNI = g->GetNI(src);
	TUNGraph::TNodeI dstNI = g->GetNI(dst);
	TIntPrV nodes;
	int srcDeg = srcNI.GetDeg();
	int dstDeg = dstNI.GetDeg();
	for (int i = 0; i < srcDeg; i++)
	{
		int srcNbr = srcNI.GetNbrNId(i);
		for (int j = 0; j < dstDeg; j++)
		{
			int dstNbr = dstNI.GetNbrNId(j);
			if (g->IsEdge(srcNbr, dstNbr) && srcNbr != dst && dstNbr != src)
			{
				TIntPr keys;
				keys.Val1 = srcNbr;
				keys.Val2 = dstNbr;
				nodes.Add(keys);
			}
		}
	}

	return beta*beta*len2 + (beta*beta*beta)*nodes.Len();
}

double linkdata::GenerateAA(PUNGraph g, TIntV nbrs)
{
	double admicadar = 0.0;
	for (int i = 0; i < nbrs.Len(); i++)
	{
		int id = nbrs[i];
		int deg = g->GetNI(id).GetDeg();
		admicadar += 1.0 / log2(deg);
	}
	return admicadar;
}

TIntFltH linkdata::PageRankValue()
{
	//TIntFltH pr;
	TSnap::GetPageRank(ugFeature, pr);

	TIntFltH corePR;
	for (TUNGraph::TNodeI NI = ugFeature->BegNI(); NI < ugFeature->EndNI(); NI++)
	{
		int k = NI.GetId();
		double v = pr.GetDat(k);
		if (core.IsKey(NI.GetId()))
		{
			corePR.AddDat(k, v);
		}
	}
	corePR.SortByDat(false);
	int percent = 10;
	int splitIndex = core.Len() * percent / 100;

	int keyUG = corePR.GetKey(splitIndex);
	ugFlag = corePR.GetDat(keyUG);
	return corePR;
}

void linkdata::GenerateKCore(int k)
{
	TIntH coreInFeatureSet;
	for (TUNGraph::TNodeI NI = ugFeature->BegNI(); NI < ugFeature->EndNI(); NI++)
	{
		if (NI.GetDeg() >= k)
		{
			coreInFeatureSet.AddKey(NI.GetId());
		}
	}

	for (TUNGraph::TEdgeI EI = ugLabel->BegEI(); EI < ugLabel->EndEI(); EI++)
	{
		int src = EI.GetSrcNId();
		int dst = EI.GetDstNId();
		bool isEdge = ugFeature->IsEdge(src, dst);

		if (coreInFeatureSet.IsKey(src) && coreInFeatureSet.IsKey(dst) && !isEdge)
		{
			core.AddKey(src);
			core.AddKey(dst);
		}
	}
	core.SortByKey();
	//// Burning changed the code in 2015-12-13-09:25
	//TIntV nodes;
	//for (int i = 0; i < core.Len(); i++)
	//{
	//	nodes.Add(core.GetKey(i));
	//}
	//ugFeature = TSnap::GetSubGraph(ugFeature, nodes);
	//net = TSnap::GetSubGraph(net, nodes);
}


TIntH linkdata::GetCore()
{
	return core;
}

linkdata::~linkdata()
{
}

void linkdata::OutputData(string path)
{
	ofstream out(path.c_str());
	size_t samples = dstDegree_m.size();
	for (size_t i = 0; i < samples; i++)
	{
		out << srcDegree_m[i] << "," << dstDegree_m[i] << ",";
		out << srcPapers_m[i] << "," << dstPapers_m[i] << ",";
		out << srcPRValue_m[i] << "," << dstPRValue_m[i] << ",";
		if (hop_m == 2)
		{
			out << cn_m[i] << "," << jc_m[i] << "," << aa_m[i] << ",";
		}
		out << katz_m[i] << ",";
		out << tagSim_m[i] << "," << affSim_m[i] << ",";
		out << propflow_m[i] << "," << sp_m[i] << "," << pa_m[i] << ",";
		out << label_m[i] << endl;
	}
	out.close();
}

void linkdata::ExtractSP()
{
	// pagerank value
	pagerank = PageRankValue();

	// search all the nodes and get the neighbors of N hops
	int count = 0;
	for (TIntH::TIter it = core.BegI(); it < core.EndI(); it++)
	{
		int srcID = it->Key;
		
		TIntV lenKNodes;
		TSnap::GetNodesAtHop(ugFeature, srcID, hop_m, lenKNodes);
		for (int i = 0; i < lenKNodes.Len(); i++)
		{
			bool isInCore = core.IsKey(lenKNodes[i]);
			bool isNotSearch = srcID < lenKNodes[i] ? 1 : 0;
			if (isInCore && isNotSearch)
			{
				count++;
				if (count % 1000 == 0)
				{
					cout << "Finished " << count << " samples" << endl;
				}
				int dstID = lenKNodes[i];

				// pagerank value
				double srcPR = pagerank.GetDat(srcID);
				srcPRValue_m.push_back(srcPR);
				double dstPR = pagerank.GetDat(dstID);
				dstPRValue_m.push_back(dstPR);

				TIntV nbrs;
				int cn = TSnap::GetCmnNbrs(ugFeature, srcID, dstID, nbrs);
				// social pattern
				int sp = GenerateSocialPattern(srcPR, dstPR, ugFlag, nbrs);
				sp_m.push_back(sp);

				// author collaboration's closeness
				int cc = GenerateCollaborationCloseness(srcID, dstID, nbrs);
				realtion_m.push_back(cc);

				// label info.
				int label = NewEdge(srcID, dstID);
				label_m.push_back(label);
			}
		}
	}
}

PNEANet linkdata::GetNet()
{
	return net;
}

PUNGraph linkdata::GetFeatureSet()
{
	return ugFeature;
}

PUNGraph linkdata::GetLabelSet()
{
	return ugLabel;
}

vector<int> linkdata::GetSP()
{
	return sp_m;
}

vector<int> linkdata::GetLabel()
{
	return label_m;
}

vector<int> linkdata::GetCC()
{
	return realtion_m;
}

vector<double> linkdata::GetAff()
{
	return affSim_m;
}

vector<double> linkdata::GetTag()
{
	return tagSim_m;
}
