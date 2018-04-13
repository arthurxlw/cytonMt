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

#include "basicHeads.h"
#include "Concatenate.h"
#include "Global.h"
#include "utils.h"

namespace cytonLib
{

Variable* Concatenate::init(string tag_, Variable* x0_, Variable* x1_)
{
	tag=tag_;
	this->x0=x0_;
	this->x1=x1_;

	assert(x0->h==1 && x0->w==1);
	assert(x1->h==1 && x1->w==1);
	assert(x0->n==x1->n);

	y.resize(x0->n, x0->c+x1->c, 1, 1);
	y.enlarge=false;
	return &y;
}

__global__
void concatenate_forward_kernel(Precision* x, int dimX, Precision* y, int dimY, bool forward)
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<dimX)
	{
		int j=blockIdx.y;
		Precision* tx=x+j*dimX+i;
		Precision* ty=y+j*dimY+i;
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

void Concatenate::forward()
{
	assert(x0->n==x1->n);
	y.resize(x0->n, x0->c+x1->c, 1, 1);

	int n=x0->n;
	int dimX0=x0->c;
	int dimX1=x1->c;
	int dimY=y.c;
	assert(dimX0 + dimX1==dimY);

	dim3 grid0(ceil(dimX0, blockSize), n);
	concatenate_forward_kernel<<<grid0, blockSize>>>(
			x0->data, dimX0, y.data, dimY, true);

	dim3 grid1(ceil(dimX1, blockSize), n);
	concatenate_forward_kernel<<<grid1, blockSize>>>(
				x1->data, dimX1, y.data+dimX0, dimY, true);
}

void Concatenate::backward()
{

	int n=y.n;
	int dimX0=x0->c;
	int dimX1=x1->c;
	int dimY=y.c;

	dim3 grid0(ceil(dimX0, blockSize), n);
	concatenate_forward_kernel<<<grid0, blockSize>>>(
			x0->grad.data, dimX0, y.grad.data, dimY, false);

	dim3 grid1(ceil(dimX1, blockSize), n);
	concatenate_forward_kernel<<<grid1, blockSize>>>(
			x1->grad.data,	dimX1, y.grad.data+dimX0, dimY, false);

}

} /* namespace cytonLib */
