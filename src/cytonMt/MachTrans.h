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

#ifndef _CYTONMT_MACHTRANS_H_
#define _CYTONMT_MACHTRANS_H_

#include "basicHeadsMt.h"
#include "ParamsMt.h"
#include "Vocabulary.h"
#include "NetworkMt.h"
#include "CorpusReader.h"

namespace cytonMt
{

class MachTrans: public cytonMt::NetworkMt
{
protected:
	XLLibTime startTime;
	XLLibTime lastCheckTime;
	int epoch;
	string bestModel;
	vector<string> savedModels;
	int numFails;
	int probePeriod;

	long int lastProbe;
	long int nSents;
	long int nSrcWords;
	long int nTrgWords;
	Precision sumTrainLikeli;

	Precision lambda;
	bool lambdaReduced;
	Precision likeliValidBest0;
	Precision likeliValidBest1;

public:
	void loadModel(string& modelDir);

	void work();

	void workTrain();

	void workTest();

	void learn(CorpusReader& corpus, bool updateParams);

	double test(CorpusReader& corpus);

	double test(vector<CorpusReader>& corpora);

	string checkTime();

	bool checkTrainStatus(bool tune=true);
};

}

#endif /* RNNTRANS_H_ */
