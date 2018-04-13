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

#include "LstmLayer.h"

namespace cytonLib
{


Variable* LstmLayer::init(string tag_, Variable* x_, bool bidirection, int hiddenSize_, int numLayers_,
		Precision dropout, cudnnRNNMode_t rnnMode_)
{
	tag=tag_;
	x=x_;
	maxSeqLen=x->n;
	batchSize=x->c;
	inputSize=x->h;
	hiddenSize=hiddenSize_;
	numLayers=numLayers_;
	bidirectFactor=bidirection?2:1;

	xDescs.init(maxSeqLen, batchSize, inputSize);

	int tHiddenSize=hiddenSize*bidirectFactor;
	y.resize(maxSeqLen, batchSize, tHiddenSize, 1);
	yDescs.init(maxSeqLen, batchSize, tHiddenSize);

	int tNumLayers = numLayers*bidirectFactor ;
	hx.resize(tNumLayers, batchSize, hiddenSize, 1);
	hx.setZero();
	cx.resize(tNumLayers, batchSize, hiddenSize, 1);
	cx.setZero();

	hy.resize(tNumLayers, batchSize, hiddenSize, 1);
	cy.resize(tNumLayers, batchSize, hiddenSize, 1);

	y.enlarge=false;

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

	// -------------------------
	// Set up the RNN descriptor
	// -------------------------

	checkError(cudnnCreateRNNDescriptor(&rnnDesc));

	rnnMode = rnnMode_;
	checkError(cudnnSetRNNDescriptor(global.cudnnHandle, rnnDesc,
			hiddenSize,
			numLayers,
			dropoutDesc,
			CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
			bidirection?CUDNN_BIDIRECTIONAL:CUDNN_UNIDIRECTIONAL,
			rnnMode,
//			CUDNN_RNN_ALGO_PERSIST_STATIC,
			CUDNN_RNN_ALGO_STANDARD,
			cudnnDataType));


	// -------------------------
	// Set up work space and reserved memory
	// -------------------------
	//	void *workspace;

	size_t workSize;
	size_t reserveSize;

	// Need for every pass
	checkError(cudnnGetRNNWorkspaceSize(global.cudnnHandle, rnnDesc, maxSeqLen, xDescs.descs, &workSize));
	// Only needed in training, shouldn't be touched between passes.
	checkError(cudnnGetRNNTrainingReserveSize(global.cudnnHandle, rnnDesc, maxSeqLen, xDescs.descs, &reserveSize));

	workspace.resize(workSize, 1);
	reserveSpace.resize(reserveSize,1);

	w.init(tag, rnnDesc, xDescs.descs[0], numLayers, bidirection, hiddenSize, rnnMode);

	checkError(cudaDeviceSynchronize());

	return &y;

}

void LstmLayer::forward()
{
	assert(x->n <= maxSeqLen);
	assert(y.c==x->c);

	y.resize(x->n, y.c, y.h, y.w);

	if(!testMode)
	{
		checkError(cudnnRNNForwardTraining(global.cudnnHandle, rnnDesc,
				x->n, xDescs.descs, x->data, hx.desc, hx.data, cx.desc,	cx.data,
				w.desc,	w.data,
				yDescs.descs, y.data, hy.desc, hy.data, cy.desc, cy.data,
				workspace.data,	workspace.ni, reserveSpace.data, reserveSpace.ni));
	}
	else
	{
		checkError(cudnnRNNForwardInference(global.cudnnHandle, rnnDesc,
				x->n, xDescs.descs, x->data, hx.desc, hx.data, cx.desc, cx.data,
				w.desc, w.data,
				yDescs.descs, y.data, hy.desc,	hy.data, cy.desc, cy.data,
				workspace.data,	workspace.ni));
	}
}

void LstmLayer::backward()
{
	assert(!testMode);
	checkError(cudnnRNNBackwardData(global.cudnnHandle, rnnDesc,
			x->n, yDescs.descs, y.data, yDescs.descs, y.grad.data, hy.desc, hy.grad.data,
			cy.desc, cy.grad.data,	w.desc,	w.data,
			hx.desc, hx.data,	cx.desc, cx.data,
			xDescs.descs,	x->grad.data,  hx.desc,	hx.grad.data, cx.desc,	cx.grad.data,
			workspace.data,	workspace.ni,	reserveSpace.data,	reserveSpace.ni ));
}

void LstmLayer::calculateGradient()
{
	assert(!testMode);
	checkError(cudnnRNNBackwardWeights( global.cudnnHandle, rnnDesc,
			x->n, xDescs.descs, x->data, hx.desc,	hx.data, yDescs.descs, y.data,
			workspace.data,	workspace.ni,	w.desc, w.grad.data,	reserveSpace.data, reserveSpace.ni ));
}




} /* namespace cytonLib */
