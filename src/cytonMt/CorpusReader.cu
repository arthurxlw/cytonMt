/*
Copyright 2018 XIAOLIN WANG (xiaolin.wang@nict.go.jp; arthur.xlw@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "CorpusReader.h"
#include "Global.h"

using namespace cytonLib;

namespace cytonMt
{

void packMatrix(vector<vector<int>>& sents, int maxLen, int batchSize,
		HostMatInt& matrix, bool src, int vocabSos, int vocabEos, int vocabEmpty)
{
	assert(sents.size()<=batchSize);
	int nj=maxLen;
	if(!src)
	{
		nj+=2;
	}
	matrix.resize(batchSize, nj);
	for(int i=0; i<batchSize; i++)
	{
		vector<int>& sent=sents.at(i%sents.size());
		int j=0;
		if(!src)
		{
			matrix.at(i, j++)= vocabSos;
		}

		for(vector<int>::iterator it=sent.begin(); it!=sent.end(); it++)
		{
			matrix.at( i, j++)= *it;
		}

		if(!src)
		{
			matrix.at( i, j++)=vocabEos;
		}

		for(;j<nj;j++)
		{
			matrix.at(i, j)= vocabEmpty;
		}
	}
}


void CorpusReader::init(const string& srcFile, const string& trgFile,
		Vocabulary& srcVocab, Vocabulary&trgVocab, bool ignoreUnk,
		int batchSize, int maxLen)
{
	this->batchSize=batchSize;
	int oldSize=nodes.size();
	nSents=0;

	//sort by source size
	vector<string> srcLines;
	XLLib::readFile(srcFile, srcLines);
	vector<string> trgLines;
	XLLib::readFile(trgFile, trgLines);
	if(srcLines.size()!=trgLines.size())
	{
		fprintf(stderr, "srcFile and trgFile have different lines:\n %s %d\n %s %d\n",
				srcFile.c_str(), srcLines.size(), trgFile.c_str(), trgLines.size());
	}

	vector<double> lens;
	vector<vector<int> > srcWhole;
	vector<vector<int> > trgWhole;
	{
		vector<int> srcSent;
		vector<int> trgSent;
		for(int idx=0;idx<srcLines.size();idx++){
			string src=srcLines.at(idx);
			srcVocab.parse(src, srcSent, ignoreUnk);

			string trg=trgLines.at(idx);
			trgVocab.parse(trg, trgSent, ignoreUnk);

			if(!srcSent.empty() && !trgSent.empty() &&  srcSent.size()<=maxLen && trgSent.size()<=maxLen)
			{
				double tLen=srcSent.size()+(double)trgSent.size()/10000.0;
				lens.push_back(tLen);
				srcWhole.push_back(srcSent);
				trgWhole.push_back(trgSent);
			}
		}
	}

	vector<int> idxs;
	XLLib::sortIndex(lens, idxs);
	vector<vector<int> > srcSents;
	vector<vector<int> > trgSents;
	for(vector<int>::iterator it=idxs.begin(); it!=idxs.end(); it++)
	{
		int idx=*it;
		vector<int>& srcSent=srcWhole.at(idx);
		vector<int>& trgSent=trgWhole.at(idx);

		if( srcSents.empty() || srcSents.back().size()==srcSent.size())
		{
			srcSents.push_back(srcSent);
			trgSents.push_back(trgSent);
			if(srcSents.size()==batchSize)
			{
				submit(srcSents, trgSents);
			}
		}
		else{
			submit(srcSents, trgSents);
			srcSents.push_back(srcSent);
			trgSents.push_back(trgSent);
		}
	}
	if(!srcSents.empty())
	{
		submit(srcSents, trgSents);
	}

//	sIdxs.clear();
	for(int i=oldSize;i<nodes.size();i++)
	{
		sIdxs.push_back(i);
	}

	XLLib::printfln("%s:%s, %d batches, %d sents", srcFile.c_str(), trgFile.c_str(),
			nodes.size()-oldSize, nSents);
}


void CorpusReader::submit(vector<vector<int> >& srcSents, vector<vector<int> >& trgSents)
{
	assert(srcSents.size()==trgSents.size());
	nSents+=srcSents.size();

	int maxSrcLen=srcSents.at(0).size();
	int maxTrgLen=0;
	for(vector<vector<int>>::iterator it=trgSents.begin(); it!=trgSents.end(); it++)
	{
		maxTrgLen=max(maxTrgLen, (int)(*it).size());
	}
	CorpusReadNode* node=new CorpusReadNode();
	nodes.push_back(node);

	{
		int tVocabSos=vocabSos;
		int tVocabEos=vocabEos;
		int tVocabEmpty=vocabEmpty;

		packMatrix(srcSents, maxSrcLen, batchSize, node->srcMat, true, tVocabSos, tVocabEos, tVocabEmpty);
		packMatrix(trgSents, maxTrgLen, batchSize, node->trgMat, false, tVocabSos, tVocabEos, tVocabEmpty);
	}

	node->factor=(Precision)srcSents.size()/batchSize;

	srcSents.clear();
	trgSents.clear();

}

void CorpusReader::reset()
{
	k=0;
	std::random_shuffle(sIdxs.begin(), sIdxs.end());
}

bool CorpusReader::read(CorpusReadNode*& node)
{
	if(k<sIdxs.size())
	{
		int idx=sIdxs.at(k);
		node=nodes.at(idx);
		k+=1;
		return true;
	}
	else
	{
		return false;
	}
}

} /* namespace cudaRnnTrans */
