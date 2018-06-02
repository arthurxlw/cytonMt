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

#ifndef _CYTONLIB_LSTMINSTANCE_H_
#define _CYTONLIB_LSTMINSTANCE_H_

#include "utils.h"
#include "DeviceMatrix.h"
#include "LstmCell.h"
#include "Variable.h"
#include "Layer.h"

namespace cytonLib
{

class LstmInstance: public Layer
{

public:
	LstmCell* cell;
	Variable* x;
	Variable* hx;
	Variable* cx;

	Variable hy;
	Variable cy;

protected:
	int seqLen;
	int numLayers;
	int batchSize;
	int inputSize;
	int hiddenSize;

	cudnnTensorDescriptor_t xDesc;
	cudnnTensorDescriptor_t yDesc;
	cudnnTensorDescriptor_t hDesc;
	cudnnTensorDescriptor_t cDesc;

	DeviceMatrix<char> reserveSpace;

public:

	Variable* init(string tag_, LstmCell* cell_, Variable* x, Variable* hx, Variable* cx);

	void prevTransfer(DevMatPrec* hx, DevMatPrec* cx, DevMatPrec* hy, DevMatPrec* cy);

	void setStateGradZero();

	void forward();

	void backward();

	void calculateGradient();

};

} /* namespace cytonLib */

#endif /* LSTMLAYER_H_ */
