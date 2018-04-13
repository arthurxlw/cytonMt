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

#include "Transpose01.h"
#include "Global.h"

namespace cytonLib
{

__global__
void transpose01_forward_kernel(Precision* x, Precision* y, int dim2, int dim1, int dim0)
{
	int i0=blockDim.x*blockIdx.x+threadIdx.x;
	if(i0<dim0)
	{
		int i2=blockIdx.z;
		int i1=blockIdx.y;
		y[i1*dim2*dim0+i2*dim0+i0]=x[i2*dim1*dim0+i1*dim0+i0];
	}
}

__global__
void transpose01_backward_kernel(Precision* x, Precision* y, int dim2, int dim1, int dim0, bool add)
{
	int i0=blockDim.x*blockIdx.x+threadIdx.x;
	if(i0<dim0)
	{
		int i2=blockIdx.z;
		int i1=blockIdx.y;
		Precision* tx=x+i2*dim1*dim0+i1*dim0+i0;
		Precision* ty=y+i1*dim2*dim0+i2*dim0+i0;
		if(add)
		{
			*tx+=*ty;
		}
		else
		{
			*tx=*ty;
		}
	}

}

Variable* Transpose01::init(string tag_, Variable* x_)
{
	tag=tag_;
	x=x_;
	y.resize(x->c, x->n, x->h, x->w);
	y.enlarge=false;
	return &y;
}

void Transpose01::forward()
{
	y.resize(x->c, x->n, x->h, x->w);

	int dim2=x->n;
	int dim1=x->c;
	int dim0=x->h*x->w;

	dim3 grid(ceil(dim0, blockSize),dim1, dim2);
	transpose01_forward_kernel<<<grid, blockSize>>>(x->data, y.data, dim2, dim1, dim0);
}

void Transpose01::backward()
{
	int dim2=x->n;
	int dim1=x->c;
	int dim0=x->h*x->w;

	dim3 grid(ceil(dim0, blockSize),dim1, dim2);
	transpose01_backward_kernel<<<grid, blockSize>>>(x->grad.data, y.grad.data, dim2, dim1, dim0, addGrad);
}

} /* namespace cytonLib */
