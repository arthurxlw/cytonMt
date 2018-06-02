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

#include "EmbeddingInstance.h"
#include <map>
#include "HostMatReal.h"
#include "cublasWrapper.h"
#include "Global.h"

namespace cytonLib {

Variable* EmbeddingInstance::init(string tag_, EmbeddingCell* cell_,
		DevMatInt* x_, HostMatInt* hx_)
{
	tag=tag_;
	cell=cell_;
	x=x_;
	hx=hx_;

	int maxLen=x->nj;
	int n=x->ni;
	int c=cell->hiddenSize;
	y.resize(maxLen, n, c, 1);
	y.enlarge=false;
	yResize=true;
	stride=cell->hiddenSize;

	deviceW.resize(1,1);

	return &y;
}

Variable* EmbeddingInstance::init(string tag_, EmbeddingCell* cell_,
		DevMatInt* x_, HostMatInt* hx_, Precision* y_, Precision* dy_, int stride)
{
	tag=tag_;
	cell=cell_;
	x=x_;
	hx=hx_;

	deviceW.resize(1,1);

	int maxLen=x->nj;
	int n=x->ni;
	int c=cell->hiddenSize;
	y.setWithStrideH(maxLen, n, c, stride, y_, dy_);
	y.enlarge=false;
	yResize=false;
	this->stride=stride;

	return &y;
}

__global__ void embeddingWeight_whole2cache_kernel(int* words, int* firstOccurs, int len, int dim, int stride,
		Precision* wholeData, Precision *cacheData, bool forward )
{
	int i=blockIdx.y;
	int j=blockDim.x*blockIdx.x+threadIdx.x;
	if(j<dim)
	{
		int word=words[i];
		Precision* tWhole=wholeData+dim*word+j;
		Precision* tCache=cacheData+stride*i+j;
		if(forward)
		{
			*tCache=*tWhole;
		}
		else{
			int firstOccur=firstOccurs[i];
			if(firstOccur<0)
			{
				*tWhole+=*tCache;
			}
		}
	}
}

void embeddingWeight_whole2cache(int* words, int* firstOccurs, int len, int dim, int stride,
		Precision* wholeData, Precision* cacheData)
{
	dim3 grid(ceil(dim, blockSize), len);
	embeddingWeight_whole2cache_kernel<<<grid, blockSize>>>(words, firstOccurs, len, dim, stride, wholeData, cacheData, true);
	checkError(cudaGetLastError());
}

void embeddingWeight_cache2whole(int* words, int* firstOccurs, int len, int dim, int stride,
		Precision* wholeData, Precision* cacheData)
{
	dim3 grid(ceil(dim, blockSize), len);
	embeddingWeight_whole2cache_kernel<<<grid, blockSize>>>(words, firstOccurs, len, dim, stride, wholeData, cacheData, false);
	checkError(cudaGetLastError());
}


void EmbeddingInstance::forward()
{
	int seqLen=x->nj;
	assert(y.c==x->ni);
	if(yResize)
	{
		y.resize(seqLen, y.c, y.h, y.w);
	}
	embeddingWeight_whole2cache(x->data, NULL, x->length(), y.h, stride,
			cell->w.data, y.data);
}


void EmbeddingInstance::apply(int word)
{
	int dim=cell->hiddenSize;
	checkError(cudaMemcpy(y.data, cell->w.data+dim*word, sizeof(Precision)*dim, cudaMemcpyDefault));
}


void EmbeddingInstance::makeFirstOccurs(int* words, int* firstOccurs, int n)
{
	std::map<int, int> dict;
	for(int i=0; i<n; i++)
	{
		int word=words[i];
		std::map<int, int>::iterator it=dict.find(word);
		if(it==dict.end())
		{
			firstOccurs[i]=-1;
			dict[word]=i;
		}
		else
		{
			firstOccurs[i]=it->second;
		}
	}
}

void EmbeddingInstance::backward()
{
	int len=x->length();
	int dim=cell->hiddenSize;
	int stride=dim;

	hFirstOccurs.resize(len, 1);
	makeFirstOccurs(hx->data, hFirstOccurs.data, len);
	firstOccurs.copyFrom(hFirstOccurs);

	for(int i=0;i<len;i++)
	{
		int firstOccur=hFirstOccurs.data[i];
		if(firstOccur>=0)
		{
			Precision* des=y.grad.data+stride*firstOccur;
			Precision* src=y.grad.data+stride*i;
			checkError(cublasXaxpy(global.cublasHandle, dim, &global.one,
						src, 1, des, 1));
			checkError(cudaMemset(src, 0, sizeof(Precision)*dim));
		}
	}
}

void EmbeddingInstance::calculateGradient()
{
	int len=x->length();
	int dim=cell->hiddenSize;
	embeddingWeight_cache2whole(x->data, firstOccurs.data, len,  dim, stride,
			cell->w.grad.data, y.grad.data);
}



} /* namespace cytonLib */
