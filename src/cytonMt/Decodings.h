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

#ifndef _CYTONMT_DECODINGS_H_
#define _CYTONMT_DECODINGS_H_

#include "utils.h"
#include "Decoding.h"
#include "Vocabulary.h"
#include "DecodingCell.h"
#include "Variable.h"
#include "Network.h"

using namespace cytonLib;

namespace cytonMt
{

class Decodings: public Network
{

public:
	EmbeddingLayer embedding;
	DecodingCell cell;

	vector<Decoding> decodings;
	vector<Variable> trgEmbeddings;

	DevMatPrec dx;

protected:
	Variable* hsSeq;
	Variable* embeddingY;

	int hiddenSize;
	int maxSeqLen1;
	DevMatInt* wordsTrg;
	Vocabulary* vocab;

	Precision *lambda;
	DevMatInt* mask;

public:
	void preInit(int maxSeqLen, string mode)
	{
		if(mode=="testBeam")
		{
			maxSeqLen1=1;
		}
		else
		{
			this->maxSeqLen1=maxSeqLen+1;
		}
		decodings.resize(maxSeqLen1);
		trgEmbeddings.resize(maxSeqLen1);
	}

	void init(Variable* hsSeq, Variable* hEncoder, Variable* cEncoder,
			DevMatInt* wordsTrg, HostMatInt* hWordsTrg,
			int vocabSize, int embedSize, int hiddenSize, int numLayers_,
			Vocabulary* vocab, Weight* embedW);

	void forward();

	Precision getScore();

	void backward();
	Precision backwardScore;

	void calculateGradient();

	void getInitState(ModelState* state);

	void transfer(ModelState* start, int word, ModelState* end, DevMatPrec& probs);
};

} /* namespace cytonLib */

#endif /* DECODINGS_H_ */
