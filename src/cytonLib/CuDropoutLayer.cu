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

#include "CuDropoutLayer.h"

namespace cytonLib
{

Variable* CuDropoutLayer::init(string tag_, Variable* x_, Precision dropout)
{
	tag=tag_;
	this->x=x_;

	localTestMode=testMode;
	if(!testMode)
	{
		y.resize(*x_);
		checkError(cudnnCreateDropoutDescriptor(&dropoutDesc));

		checkError(cudnnDropoutGetStatesSize(global.cudnnHandle, &stateSize));
		checkError(cudaMalloc((void **)&stateSpace, stateSize));

		checkError(cudnnSetDropoutDescriptor(dropoutDesc, global.cudnnHandle,
				dropout, stateSpace, stateSize, global.rnnDropoutSeed++));

		checkError(cudnnDropoutGetReserveSpaceSize(x_->desc,	&reserveSize));
		checkError(cudaMalloc((void **)&reserveSpace, reserveSize));
	}
	else
	{
		y.set(*x);
	}

	return &y;
}

void CuDropoutLayer::forward()
{
	if(!localTestMode)
	{
		if(!cytonLib::testMode)
		{
			y.resize(*x);
			checkError(cudnnDropoutForward(
					global.cudnnHandle, dropoutDesc,
					x->desc, x->data, y.desc, y.data,
					reserveSpace, reserveSize));
		}
		else
		{
			y.copyFrom(*x);
		}
	}
}

void CuDropoutLayer::backward()
{
	assert(!localTestMode && !cytonLib::testMode);
	checkError(cudnnDropoutBackward(
			global.cudnnHandle, dropoutDesc,
			y.desc, y.grad.data, x->desc, x->grad.data,
			reserveSpace, reserveSize));

}

} /* namespace reinLearnSentSeg */
