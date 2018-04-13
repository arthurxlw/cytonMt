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

#include "CudnnDescriptors.h"

namespace cytonLib
{


void CudnnDescriptors::createNdDesc(cudnnTensorDescriptor_t& desc, int d0, int d1, int d2)
{

	int dim[3];
	dim[0]=d0;
	dim[1]=d1;
	dim[2]=d2;

	int stride[3];
	stride[0]=dim[1]*dim[2];
	stride[1]=dim[2];
	stride[2]=1;

	checkError(cudnnCreateTensorDescriptor(&desc));
	checkError(cudnnSetTensorNdDescriptor(desc, cudnnDataType, 3, dim, stride));
}

CudnnDescriptors::CudnnDescriptors()
{
	maxLen=0;
	n=0;
	c=0;
	h=0;
	w=0;
	descs=NULL;

}

void CudnnDescriptors::init(int maxLen_, int n_, int c_, int h_, int w_)
{
	maxLen=maxLen_;
	n=n_;
	c=c_;
	h=h_;
	w=w_;
	descs=new cudnnTensorDescriptor_t[maxLen];

	for(int i=0; i<maxLen; i++)
	{
		cudnnTensorDescriptor_t& desc=descs[i];

		assert(w==1);

		createNdDesc(desc, n, c, h);
	}
}

CudnnDescriptors::~CudnnDescriptors()
{
	for(int i=0; i<maxLen; i++)
	{
		cudnnTensorDescriptor_t& desc=descs[i];
		checkError(cudnnDestroyTensorDescriptor(desc));
	}
	delete []descs;
}


} /* namespace cytonVR */
