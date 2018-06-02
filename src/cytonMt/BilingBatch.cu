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

#include "BilingBatch.h"
#include "Vocabulary.h"

namespace cytonMt
{

void BilingBatch::init(int batchSize_, int maxSeqLen_)
{
	this->batchSize=batchSize_;
	this->maxSeqLen=maxSeqLen_;

	this->srcLen=maxSeqLen;
	this->trgLen=maxSeqLen+2;

	srcMat.resize(batchSize, srcLen);
	srcMat.setZero();
	trgMat.resize(batchSize, trgLen);
	trgMat.setZero();
}

void BilingBatch::setSrc(HostMatInt& hSrcMat_)
{
	hSrcMat.set(hSrcMat_);
	assert(hSrcMat.ni==batchSize);
	srcMat.copyFrom(hSrcMat);
	srcLen=srcMat.nj;
}


void BilingBatch::setSrcTrg(HostMatInt& hSrcMat_, HostMatInt& hTrgMat_, Precision factor)
{
	setSrc(hSrcMat_);

	hTrgMat.set(hTrgMat_);
	assert(hTrgMat.ni==batchSize);
	trgMat.copyFrom(hTrgMat);
	trgLen=trgMat.nj;

	this->factor=factor;
}

int BilingBatch::numTrgWords()
{
	int numTrgWords=0;
	for(int k=0; k<hTrgMat.length();k++)
	{
		if(hTrgMat.at(k)!=vocabEmpty)
		{
			numTrgWords+=1;
		}
	}
	numTrgWords-=batchSize;
	return numTrgWords;
}

} /* namespace cytonLib */
