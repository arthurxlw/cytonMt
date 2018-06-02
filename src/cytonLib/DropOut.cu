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

#include "DropOut.h"

namespace cytonLib
{

Variable* DropOut::init(string tag_, Variable* x, Precision dropout)
{
	tag=tag_;
	this->x=x;
	this->dropout=dropout;
	this->active=dropout>1e-6;
	this->localTestMode=testMode;
	if(active && ! localTestMode)
	{
		y.resize(*x);
		y.enlarge=false;
	}
	else
	{
		y.set(*x);
	}
	return &y;
}

__global__
void dropOut_forward_kernel(Precision* src, Precision* mask, Precision* des, Precision dropOut, int len)
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<len)
	{
		if(mask[i]<dropOut)
		{
			des[i]=0;
		}
		else
		{
			des[i]=src[i]/(1-dropOut);
		}

	}
}


__global__
void dropOut_backward_kernel(Precision* src, Precision* mask, Precision* des, Precision dropOut, int len)
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<len)
	{
		if(mask[i]<dropOut)
		{
			des[i]=0;
		}
		else
		{
			des[i]=src[i]/(1-dropOut);
		}

	}
}

void DropOut::forward()
{

	if(active && !localTestMode)
	{
		if(!testMode)
		{
			mask.resize(x->ni, x->nj);
			mask.initRandomCurand();
			y.resize(*x);

			int len=y.length();
			assert(mask.length()==len);
			dropOut_forward_kernel<<<ceil(len,blockSize),blockSize>>>(x->data, mask.data, y.data,
					dropout, len);
		}
		else
		{
			y.copyFrom(*x);
		}

	}
	else
	{
		y.set(*x);
	}
}


void DropOut::backward()
{
	if(active)
	{
		assert(!testMode);

		int len=x->length();
		assert(mask.length()==len);
		dropOut_backward_kernel<<<ceil(len,blockSize),blockSize>>>(y.grad.data, mask.data, x->grad.data,
				dropout, len);
	}
}


} /* namespace cytonLib */
