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

#ifndef _CYTONLIB_LSTMLAYER_H_
#define _CYTONLIB_LSTMLAYER_H_

#include "utils.h"
#include "DeviceMatrix.h"
#include "Weight.h"
#include "LstmWeight.h"
#include "Variable.h"
#include "CudnnDescriptors.h"
#include "Layer.h"

namespace cytonLib
{

class LstmLayer: public Layer
{
public:
	Variable* x;
	Variable y;

	LstmWeight w;
	Variable hx;
	Variable cx;
	Variable hy;
	Variable cy;

protected:

	int bidirectFactor;

	CudnnDescriptors xDescs;
	CudnnDescriptors yDescs;

	DeviceMatrix<char> workspace;
	DeviceMatrix<char> reserveSpace;

	cudnnRNNDescriptor_t rnnDesc;
	cudnnRNNMode_t rnnMode;
	cudnnDropoutDescriptor_t dropoutDesc;

	int maxSeqLen;
	int numLayers;
	int batchSize;
	int inputSize;
	int hiddenSize;

public:
	Variable* init(string tag_, Variable* x_, bool bidirection, int hiddenSize_, int numLayers_,
			Precision dropout, cudnnRNNMode_t rnnMode_=CUDNN_LSTM);

	void forward();

	void backward();

	void calculateGradient();
};

} /* namespace cytonLib */

#endif /* LSTMLAYER_H_ */
