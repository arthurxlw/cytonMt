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

#ifndef _CYTONMT_PARAMSMT_H_
#define _CYTONMT_PARAMSMT_H_

#include "basicHeadsMt.h"
#include "ParametersBase.h"
#include "basicHeads.h"

using namespace cytonLib;

namespace cytonMt
{


class ParamsMt: public xllib::ParametersBase
{
public:
	string mode;
	string saveModel;
	string loadModel;
	int maxSaveModels;
	string trainData;
	string devData;
	string testInput;
	string testOutput;
	string srcVocab;
	string trgVocab;
	int srcVocabSize;
	int trgVocabSize;
	bool ignoreUnk;

	Precision initParam;
	string optimization;
	Precision learningRate;
	Precision decayRate;
	int decayStart;
	bool decayConti;
	bool decayStatus;

	int epochs;
	int epochStart;
	vector<int> hiddenSize;
	int maxSeqLen;
	int numLayers;
	Precision dropout;
	Precision clipGradient;
	Precision labelSmooth;
	double probeFreq;
	Precision probeMargin;
	int patience;

	int beamSize;
	Precision lenPenalty;

	ParamsMt();

	void init_members();

	void saveModelParams(std::string fileName);
	void loadModelParams(std::string fileName);

};


extern ParamsMt params;

};

#endif /* SRC_SIMINTERPRETSEGMENT_PARAMETERS_H_ */
