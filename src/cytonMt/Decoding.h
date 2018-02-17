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

#ifndef _CYTONMT_DECODING_H_
#define _CYTONMT_DECODING_H_

#include "utils.h"
#include "EmbeddingInstance.h"
#include "Attention.h"
#include "Concatenate.h"
#include "LstmInstance.h"
#include "SoftmaxLayer.h"
#include "CuDropoutLayer.h"
#include "Variable.h"
#include "DecodingCell.h"
#include "ModelState.h"
#include "EmbeddingLayer.h"
#include "ActivationLayer.h"
#include "DuplicateLayer.h"
#include "Network.h"


using namespace cytonLib;

namespace cytonMt
{

class Decoding: public Network
{
public:

	int index;

	Variable* embeddingY;
	Concatenate concateOutEmbed;

	LstmInstance lstm;

	Attention attention;

	CuDropoutLayer dropHo;
	DuplicateLayer dupHo;
	LinearLayer linearHo;

	SoftmaxLayer softmax;

protected:
	int vocabSize;

public:

	void init(int index, DecodingCell& cell, Variable* embeddingY, Variable* hs,
			Decoding* prev, int hiddenSize);

	void backwardPrepare(DevMatInt* wordTrg, int offset,  Precision scale);

	void transfer(ModelState* start, ModelState* end, DevMatPrec& probs);

};

} /* namespace cytonLib */

#endif /* DECODING_H_ */
