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

#ifndef _CYTONMT_BEAMSEARCH_H_
#define _CYTONMT_BEAMSEARCH_H_

#include "basicHeads.h"
#include "NetworkMt.h"
#include "ModelState.h"
#include "BilingBatch.h"
#include "Factory.h"
#include "Stack.h"


namespace cytonLib
{

class BeamSearch
{
	cytonMt::NetworkMt* model;
	cytonMt::Vocabulary* vocab;
	int beamSize;
	Precision lenPenalty;
	int maxSeqLen1;
	ModelState initModelState;
	Factory<ModelState> modelStateFactory;

	StackState initStackState;
	Factory<StackState> stackStateFactory;
	Stack stack;
	DevMatPrec probs;

public:
	void init(cytonMt::NetworkMt* model_, cytonMt::Vocabulary *vocab_,
			int beamSize_, int maxSeqLen_, Precision lenPenalty,
			vector<int>& hiddenSize, int numLayers);

	string apply(vector<string>& trans, Precision* score=NULL);

	string getDerivation(StackState* state, vector<string>& words);

};

} /* namespace cytonLib */

#endif /* BEAMSEARCH_H_ */
