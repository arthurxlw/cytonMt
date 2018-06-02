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

#ifndef _CYTONMT_ENCODER_H_
#define _CYTONMT_ENCODER_H_

#include "LstmLayer.h"
#include "LstmStateBidir2mono.h"
#include "Transpose01.h"
#include "Network.h"

using namespace cytonLib;

namespace cytonMt
{

class Encoder: public Network
{
public:
	LstmLayer lstm;
	LstmStateBidir2mono hyMono;
	LstmStateBidir2mono cyMono;
	Transpose01 transY;

	Variable* y;
	Variable* hy;
	Variable* cy;

	Variable* init(Variable* x, int maxSeqLen, int batchSize, int numLayers,
			int embedSize, int hiddenSize);
};

} /* namespace cytonLib */

#endif /* ENCODER_H_ */
