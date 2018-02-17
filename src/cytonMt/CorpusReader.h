/*
Copyright 2018 XIAOLIN WANG (xiaolin.wang@nict.go.jp; arthur.xlw@google.com)

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

#ifndef _CYTONMT_CORPUSREADER_H_
#define _CYTONMT_CORPUSREADER_H_

#include "basicHeads.h"
#include "Vocabulary.h"
#include "HostMatrix.h"

using namespace cytonLib;

namespace cytonMt
{

class CorpusReadNode
{
public:
	HostMatInt srcMat;
	HostMatInt trgMat;
	Precision factor;

	CorpusReadNode(): factor(0.0)
	{
	}
};


class CorpusReader
{
	int batchSize;
	int k;
public:

	vector<CorpusReadNode*> nodes;
	vector<int> sIdxs;
	int nSents;

	void init(const string& srcFile, const string& trgFile,
			Vocabulary& srcVocab, Vocabulary&trgVocab, bool ignoreUnk,
			int batchSize, int maxLen);

	void submit(vector<vector<int> >& srcSents, vector<vector<int> >& trgSents);

	void reset();

	bool read(CorpusReadNode*& node);

};

void packMatrix(vector<vector<int>>& sents, int maxLen, int batchSize,
		HostMatInt& matrix, bool src, int vocabSos, int vocabEos, int vocabEmpty);

} /* namespace cudaRnnTrans */

#endif /* CORPUSREADER_H_ */
