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

#include "LstmInstance.h"
#include "Global.h"
#include "CudnnDescriptors.h"

namespace cytonLib
{

Variable* LstmInstance::init(string tag_, LstmCell* cell_,
		Variable* x_, Variable* hx_, Variable *cx_)
{
	tag=tag_;
	cell=cell_;
	x=x_;
	hx=hx_;
	cx=cx_;

	batchSize=cell->batchSize;
	inputSize=cell->inputSize;
	hiddenSize=cell->hiddenSize;
	numLayers=cell->numLayers;
	seqLen=1;

	assert(x->n==batchSize && x->c == inputSize);
	assert(hx->n==numLayers && hx->c==batchSize && hx->h == hiddenSize);
	assert(cx->n==numLayers && cx->c==batchSize && cx->h == hiddenSize);

	y.resize(batchSize, hiddenSize, 1, 1);
	hy.resize(numLayers, batchSize, hiddenSize, 1);
	cy.resize(numLayers, batchSize, hiddenSize, 1);


	CudnnDescriptors::createNdDesc(xDesc, x->n, x->c, x->h);
	CudnnDescriptors::createNdDesc(yDesc, y.n, y.c, y.h);
	CudnnDescriptors::createNdDesc(hDesc, hx->n, hx->c, hx->h);
	CudnnDescriptors::createNdDesc(cDesc, cx->n, cx->c, cx->h);

	// Only needed in training, shouldn't be touched between passes.
	size_t reserveSize;
	checkError(cudnnGetRNNTrainingReserveSize(global.cudnnHandle, cell->rnnDesc, seqLen, &x->desc, &reserveSize));
	reserveSpace.resize(reserveSize, 1);

	checkError(cudaDeviceSynchronize());
	return &y;
}

void LstmInstance::prevTransfer(DevMatPrec* hx_, DevMatPrec* cx_, DevMatPrec* hy_, DevMatPrec* cy_)
{
	hx->data=hx_->data;
	cx->data=cx_->data;
	hy.data=hy_->data;
	cy.data=cy_->data;
}

void LstmInstance::forward()
{
	assert(seqLen==1);
	if(!testMode)
	{
		checkError(cudnnRNNForwardTraining(global.cudnnHandle, cell->rnnDesc,
						seqLen, &xDesc, x->data, hDesc, hx->data, cDesc, cx->data,
						cell->w.desc,	cell->w.data,
						&yDesc, y.data, hDesc, hy.data, cDesc, cy.data,
						cell->workspace.data,	cell->workspace.ni, reserveSpace.data, reserveSpace.ni));
	}
	else
	{
		checkError(cudnnRNNForwardInference(global.cudnnHandle, cell->rnnDesc,
					seqLen, &xDesc, x->data, hDesc, hx->data, cDesc, cx->data,
					cell->w.desc, cell->w.data,
					&yDesc, y.data, hDesc, hy.data, cDesc, cy.data,
					cell->workspace.data,	cell->workspace.ni));
	}
}

void LstmInstance::setStateGradZero()
{
	hy.grad.setZero();
	cy.grad.setZero();
}

void LstmInstance::backward()
{
	assert(!testMode);
	checkError(cudnnRNNBackwardData(global.cudnnHandle, cell->rnnDesc,
			seqLen, &yDesc, y.data, &yDesc, y.grad.data,
			hDesc, hy.grad.data,	cDesc, cy.grad.data,	cell->w.desc,	cell->w.data,
			hDesc, hx->data,	cDesc, cx->data, &xDesc, x->grad.data,
			hDesc, hx->grad.data, cDesc, cx->grad.data,
			cell->workspace.data,	cell->workspace.ni,	reserveSpace.data,	reserveSpace.ni ));

}

void LstmInstance::calculateGradient()
{
	checkError(cudnnRNNBackwardWeights(global.cudnnHandle, cell->rnnDesc,
			seqLen, &xDesc, x->data, hDesc, hx->data, &yDesc, y.data,
			cell->workspace.data,	cell->workspace.ni,
			cell->w.desc, cell->w.grad.data,
			reserveSpace.data, reserveSpace.ni ));
}

} /* namespace cytonLib */
