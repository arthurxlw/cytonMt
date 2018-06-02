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

#include "LstmCell.h"
#include "CudnnDescriptors.h"

namespace cytonLib
{

void LstmCell::init(string tag,  int batchSize_, int inputSize_, int hiddenSize_,
		int numLayers_, int maxSeqLen_, Precision dropout,
		Variable* hx_, Variable* cx_)
{
	this->batchSize=batchSize_;
	this->inputSize=inputSize_;
	this->hiddenSize=hiddenSize_;
	this->numLayers=numLayers_;
	this->hx=hx_;
	this->cx=cx_;

	assert(hx->n==numLayers && hx->c==batchSize && hx->h==hiddenSize);
	assert(cx->n==numLayers && cx->c==batchSize && cx->h==hiddenSize);
	// -------------------------
	// Set up the dropout descriptor (needed for the RNN descriptor)
	// -------------------------

	checkError(cudnnCreateDropoutDescriptor(&dropoutDesc));

	// How much memory does dropout need for states?
	// These states are used to generate random numbers internally
	// and should not be freed until the RNN descriptor is no longer used
	size_t stateSize;
	void *states;
	checkError(cudnnDropoutGetStatesSize(global.cudnnHandle, &stateSize));

	checkError(cudaMalloc(&states, stateSize));

	Precision tDropOut=dropout;
	checkError(cudnnSetDropoutDescriptor(dropoutDesc,
			global.cudnnHandle,
			tDropOut,
			states,
			stateSize,
			global.rnnDropoutSeed++));

	checkError(cudnnCreateRNNDescriptor(&rnnDesc));

	//		if      (mode == 0) RNNMode = CUDNN_RNN_RELU;
	//		else if (mode == 1) RNNMode = CUDNN_RNN_TANH;
	//		else if (mode == 2) RNNMode = CUDNN_LSTM;
	//		else if (mode == 3) RNNMode = CUDNN_GRU;

	RNNMode = CUDNN_LSTM;

	checkError(cudnnSetRNNDescriptor(global.cudnnHandle, rnnDesc,
			hiddenSize_,
			numLayers_,
			dropoutDesc,
			CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
			CUDNN_UNIDIRECTIONAL,
			RNNMode,
			CUDNN_RNN_ALGO_STANDARD,
//			CUDNN_RNN_ALGO_PERSIST_STATIC,
			cudnnDataType));


	{
		xDescs.init(maxSeqLen_, batchSize_, inputSize_);

		w.init(tag, rnnDesc, xDescs.descs[0] ,numLayers_,false, hiddenSize_);

		size_t workSize;
		checkError(cudnnGetRNNWorkspaceSize(global.cudnnHandle, rnnDesc, maxSeqLen_, xDescs.descs, &workSize));
		workspace.resize(workSize, 1);
	}

	checkError(cudaDeviceSynchronize());
}



} /* namespace cytonLib */
