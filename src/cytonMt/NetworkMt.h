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

#ifndef _CYTONMT_TRANSLATOR_H_
#define _CYTONMT_TRANSLATOR_H_


#include "ParamsMt.h"
#include "BilingBatch.h"
#include "EmbeddingLayer.h"
#include "Encoder.h"
#include "Decodings.h"
#include "ModelState.h"
#include "ModelOutput.h"
#include "CorpusReader.h"
#include "Network.h"

namespace cytonMt
{

class NetworkMt: Network
{
public:
	BilingBatch batch;
	vector<int> srcBatches;
	EmbeddingLayer embeddingSrc;
	Encoder encoder;
	Decodings decodings;

protected:
	Vocabulary srcVocab;
	Vocabulary trgVocab;
	bool shareSrcTrgEmbed;
	int nBatches;

	vector<CorpusReader> validCorpora;

public:
	void init();

	void backward();

	Precision train(Precision lambda, bool updateParams);

	Precision getScore();

	void saveModel(string fileName);

	void loadModel(string fileName);

	void prevApply();

	string apply(vector<string>& words, Precision& likeli);

	void getInitState(ModelState* state);

	void prevBeamSearch();

	void beamSearchTransfer(ModelState* start, int input,
			ModelState* end, DevMatPrec& outputs);
};

}
#endif /* TRANSLATOR_H_ */
