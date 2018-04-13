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

#include "WeightedHs.h"
#include "Global.h"
#include "cublasWrapper.h"

namespace cytonMt
{

Variable* WeightedHs::init(string tag_, Variable* x_, Variable* wx_)
{
	tag=tag_;
	x=x_;
	wx=wx_;

	int batchSize=x->n;
	int seqLen=x->c;
	int dim=x->h;
	assert(wx->n == batchSize && wx->c==seqLen);

	y.resize(batchSize, dim, 1, 1);
	return &y;

}

void WeightedHs::forward()
{
	int batchSize=x->n;
	int seqLen=x->c;
	int dim=x->h;
	assert(wx->n == batchSize && wx->c==seqLen);
	y.resize(batchSize, dim, 1, 1);

	checkError(cublasXgemmBatch(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim, 1, seqLen,
			&global.one, x->data, dim, dim*seqLen, wx->data, seqLen, seqLen,
			&global.zero, y.data, dim, dim, batchSize));
}

void WeightedHs::backward()
{
	int batchSize=x->n;
	int seqLen=x->c;
	int dim=x->h;
	assert(wx->n == batchSize && wx->c==seqLen);

	checkError(cublasXgemmBatch(global.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, seqLen, 1, dim,
				&global.one, x->data, dim, dim*seqLen, y.grad.data, dim, dim,
				&global.zero, wx->grad.data, seqLen, seqLen, batchSize));

	checkError(cublasXgemmBatch(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim, seqLen, 1,
				&global.one, y.grad.data, dim, dim, wx->data, 1, seqLen,
				&global.one, x->grad.data, dim, dim*seqLen, batchSize));
}

} /* namespace cytonLib */
