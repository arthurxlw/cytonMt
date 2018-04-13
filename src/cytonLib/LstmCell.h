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

#ifndef _CYTONLIB_LSTMCELL_H_
#define _CYTONLIB_LSTMCELL_H_

#include "utils.h"
#include "DeviceMatrix.h"
#include "LstmWeight.h"
#include "Weight.h"
#include "Variable.h"
#include "CudnnDescriptors.h"

namespace cytonLib
{

class LstmCell
{

public:
	int batchSize;
	int inputSize;
	int hiddenSize;
	int numLayers;

	LstmWeight w;

	cudnnRNNDescriptor_t rnnDesc;

	Variable* hx;
	Variable* cx;
	DeviceMatrix<char> workspace;
	CudnnDescriptors xDescs;
protected:
	cudnnRNNMode_t RNNMode;
	cudnnDropoutDescriptor_t dropoutDesc;

public:
	void init(string tag, int batchSize, int inputSize, int hiddenSize,	int numLayers,	int maxSeqLen, Precision dropout,
			Variable* hxBegin, Variable* cxBegin);
};

} /* namespace cytonLib */

#endif /* LSTM_H_ */
