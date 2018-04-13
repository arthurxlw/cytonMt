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

#include "LstmStateBidir2mono.h"
#include "Global.h"

namespace cytonLib
{

void LstmStateBidir2mono::init(string tag_, Variable* x_)
{
	tag=tag_;
	this->x=x_;
	assert(x->n%2==0);
	this->numLayers=x->n/2;
	this->batchSize=x->c;
	this->hiddenSize=x->h;

	y.resize(numLayers, batchSize, hiddenSize*2, 1);
	y.frozen=true;
}

__global__
void lstmStateBidir2mono_kernel(Precision* x, Precision* y, int batchSize, int hiddenSize,
		bool forward)
{
	int layer=blockIdx.z;
	int batch=blockIdx.y;
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<hiddenSize)
	{
		int offset=hiddenSize*batchSize*2*layer;
		for(int part=0; part<2; part++)
		{
			Precision* tx=x+offset+hiddenSize*(batchSize*part+batch)+i;
			Precision* ty=y+offset+hiddenSize*2*batch+hiddenSize*part+i;
			if(forward)
			{
				*ty=*tx;
			}
			else
			{
				*tx=*ty;
			}
			}

	}
}
void LstmStateBidir2mono::forward()
{
	dim3 grid(ceil(hiddenSize, blockSize), batchSize, numLayers);
	lstmStateBidir2mono_kernel<<<grid, blockSize>>>(x->data, y.data, batchSize, hiddenSize, true);
}

void LstmStateBidir2mono::backward()
{
	dim3 grid(ceil(hiddenSize, blockSize), batchSize, numLayers);
	lstmStateBidir2mono_kernel<<<grid, blockSize>>>(x->grad.data, y.grad.data, batchSize, hiddenSize, false);
}

} /* namespace cytonLib */
