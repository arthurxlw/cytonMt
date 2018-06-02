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

#include "LinearLayer.h"
#include "WeightFactory.h"
#include "Global.h"
#include "cublasWrapper.h"

namespace cytonLib
{

Variable* LinearLayer::init(const string& tag_, Variable* x_, int dimOutput_,
		bool biased_, int mode_, WeightFactory* weightFactory_, Weight* weight_)
{
	this->tag=tag_;
	this->x=x_;
	this->base=NULL;
	dimOutput=dimOutput_;
	biased=biased_;
	mode=mode_;

	WeightFactory* pWF=weightFactory_;
	if(pWF==NULL)
	{
		pWF=&weightFactory;
	}

	if(mode==1)
	{
		dimInput=x->c * x->h * x->w;
		y.resize(x->n, dimOutput, 1, 1);
	}
	else if(mode==2)
	{
		dimInput=x->h * x->w;
		y.resize(x->n, x->c, dimOutput, 1);
	}
	else
	{
		assert(false);
	}
	y.enlarge=false;

	pWF->create(w, tag+".w", dimInput, dimOutput, weight_);

	if(biased)
	{
		pWF->create(b, tag+".b", dimOutput, 1);
	}

	XLLib::printfln(global.os, "%s linearLayer.init xDim %s yDim %s", tag.c_str(),
				x->toStringDim().c_str(), y.toStringDim().c_str());
	return &y;
}

Variable* LinearLayer::init(const string& tag_, LinearLayer* base_, Variable* x_)
{
	tag=tag_;
	x=x_;
	base=base_;
	addGrad=base->addGrad;
	dimOutput=base->dimOutput;
	biased=base->biased;
	mode=base->mode;

	if(mode==1)
	{
		dimInput=x->c * x->h * x->w;
		assert(dimInput==base->dimInput);
		y.resize(x->n, dimOutput, 1, 1);
	}
	else if(mode==2)
	{
		dimInput= x->h * x->w;
		assert(dimInput==base->dimInput);
		y.resize(x->n, x->c, dimOutput, 1);
	}
	y.enlarge=false;

//	XLLib::printfln(global.os, "%s linearLayer.init from base, xDim %s yDim %s", tag.c_str(),
//				x->toStringDim().c_str(), y.toStringDim().c_str());

	return &y;

}

void LinearLayer::forward()
{
	if(base!=NULL)
	{
		w.set(base->w);
		if(biased)
		{
			b.set(base->b);
		}
		base=NULL;
	}
	if(mode==1)
	{
		num=x->n;
		assert(x->c*x->h*x->w == dimInput);
		y.resize(x->n, dimOutput, 1, 1);
	}
	else if(mode==2)
	{
		num=x->n*x->c;
		assert(x->h*x->w == dimInput);
		y.resize(x->n, x->c, dimOutput, 1, 1);
	}
	else
	{
		assert(false);
	}

	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			dimOutput, num, dimInput,
			&global.one, w.data, dimInput, x->data, dimInput,
			&global.zero, y.data, dimOutput));
	if(biased)
	{
		checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				dimOutput, num, 1,
				&global.one, b.data, dimOutput, global.ones(num), 1,
				&global.one, y.data, dimOutput));
	}

}

void LinearLayer::increase()
{
	if(base!=NULL)
	{
		w.set(base->w);
		if(biased)
		{
			b.set(base->b);
		}
		base=NULL;
	}

	if(mode==1)
	{
		num=x->n;
		assert(x->c*x->h*x->w == dimInput);
		y.resize(x->n, dimOutput, 1, 1);
	}
	else if(mode==2)
	{
		num=x->n*x->c;
		assert(x->h*x->w == dimInput);
		y.resize(x->n, x->c, dimOutput, 1, 1);
	}
	else
	{
		assert(false);
	}


	num=1;
	Precision* y1=y.data+y.length()-dimOutput;
	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			dimOutput, num, dimInput,
			&global.one, w.data, dimInput, x->data+x->length()-dimInput, dimInput,
			&global.zero, y1, dimOutput));

	if(biased)
	{
		checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				dimOutput, num, 1,
				&global.one, b.data, dimOutput, global.ones(num), 1,
				&global.one, y1, dimOutput));
	}


}

void LinearLayer::backward()
{
	Precision* beta=addGrad?&global.one:&global.zero;
	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			dimInput, num, dimOutput,
			&global.one, w.data, dimInput, y.grad.data, dimOutput,
			beta, x->grad.data, dimInput));
}



void LinearLayer::calculateGradient()
{
	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			dimInput, dimOutput,  num,
			&global.one, x->data, dimInput, y.grad.data, dimOutput,
			&global.one, w.grad.data, dimInput));

	if(biased)
	{
		checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				dimOutput, 1, num,
				&global.one, y.grad.data, dimOutput, global.ones(num), num,
				&global.one, b.grad.data, dimOutput));
	}


}


} /* namespace cytonLib */
