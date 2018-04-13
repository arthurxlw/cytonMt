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

#include "LstmWeight.h"
#include "Global.h"
#include "WeightFactory.h"

namespace cytonLib
{

void LstmWeight::init(string tag_, cudnnRNNDescriptor_t rnnDesc_, cudnnTensorDescriptor_t xDesc_,
		int numLayers_, bool bidirect_, int hiddenSize_, cudnnRNNMode_t rnnMode_)
{
	this->tag=tag_;
	this->rnnDesc=rnnDesc_;
	this->xDesc=xDesc_;
	this->numLayers=numLayers_;
	this->bidirect=bidirect_;
	this->hiddenSize=hiddenSize_;
	this->rnnMode=rnnMode_;

	size_t weightsSize;
	checkError(cudnnGetRNNParamsSize(global.cudnnHandle, rnnDesc, xDesc, &weightsSize,
			cudnnDataType));
	assert(weightsSize%sizeof(Precision)==0);
	int tLen = weightsSize/sizeof(Precision);

	weightFactory.create(*this, tag, tLen, 1);

	checkError(cudnnCreateFilterDescriptor(&desc));
	int dimA[3];
	dimA[0]=tLen;
	dimA[1]=1;
	dimA[2]=1;
	checkError(cudnnSetFilterNdDescriptor(desc, cudnnDataType, CUDNN_TENSOR_NCHW, 3, dimA));
}

} /* namespace cytonLib */
