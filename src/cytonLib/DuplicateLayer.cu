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

#include "DuplicateLayer.h"
#include "cublasWrapper.h"

namespace cytonLib
{

Variable* DuplicateLayer::init(string tag_, Variable* x_)
{
	tag=tag_;
	x=x_;
	localTestMode=cytonLib::testMode;

	if(!localTestMode)
	{
		y.resize(*x);
		y1.resize(*x);
	}
	else
	{
		y.set(*x);
		y1.set(*x);
	}

	return &y;
}

void DuplicateLayer::forward()
{
	if(!localTestMode)
	{
		y.copyFrom(*x);
		y1.copyFrom(*x);
	}
	else
	{
		y.set(*x);
		y1.set(*x);
	}
}

void DuplicateLayer::backward()
{
	assert(!localTestMode);
	int n=x->length();
	assert(y.length() == n && y1.length()== n );
	checkError(cudaMemcpy(x->grad.data, y.grad.data,  sizeof(Precision)*n, cudaMemcpyDefault));
	checkError(cublasXaxpy(global.cublasHandle, n, &global.one, y1.grad.data, 1, x->grad.data, 1));
}

} /* namespace cytonLib */
