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

#include "MultiplyHsHt.h"
#include "Global.h"
#include "cublasWrapper.h"
#include "ParamsMt.h"

namespace cytonMt
{

extern ParamsMt params;

Variable* MultiplyHsHt::init(string tag_, Variable* hs_, Variable* ht_)
{
	tag=tag_;
	hs=hs_;
	ht=ht_;

	int batchSize=hs->n;
	int seqLen=hs->c;
	int hiddenSize=hs->h;
	assert(ht->n == batchSize && ht->c==hiddenSize);
	y.resize(batchSize, seqLen, 1, 1);
	return &y;
}


void MultiplyHsHt::forward()
{
	int batchSize=hs->n;
	int seqLen=hs->c;
	int hiddenSize=hs->h;
	assert(ht->n == batchSize && ht->c==hiddenSize);

	y.resize(batchSize, seqLen, 1, 1);
	checkError(cublasXgemmBatch(global.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, seqLen, 1, hiddenSize,
			&global.one, hs->data, hiddenSize, hiddenSize*seqLen, ht->data, hiddenSize, hiddenSize,
			&global.zero, y.data, seqLen, seqLen, batchSize));

}

void MultiplyHsHt::backward()
{
	int batchSize=hs->n;
	int seqLen=hs->c;
	int hiddenSize=hs->h;
	assert(ht->n == batchSize && ht->c==hiddenSize);

	//backward on ht
	checkError(cublasXgemmBatch(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, hiddenSize, 1, seqLen,
						&global.one, hs->data, hiddenSize, hiddenSize*seqLen, y.grad.data, seqLen, seqLen,
						&global.zero, ht->grad.data, hiddenSize, hiddenSize, batchSize));

	checkError(cublasXgemmBatch(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, hiddenSize, seqLen, 1,
							&global.one, ht->data, hiddenSize, hiddenSize, y.grad.data, 1, seqLen,
							&global.one, hs->grad.data, hiddenSize, hiddenSize*seqLen, batchSize));

}



} /* namespace cytonMt */
