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

#include "Decodings.h"
#include "Global.h"
#include "ParamsMt.h"

namespace cytonMt
{

void Decodings::init(Variable* hsSeq, Variable* hEncoder, Variable* cEncoder,
		DevMatInt* wordsTrg, HostMatInt* hWordsTrg,
		int vocabSize, int embedSize, int hiddenSize, int numLayers_,
		Vocabulary* vocab, Weight* embedW)
{
	this->hsSeq=hsSeq;
	this->wordsTrg=wordsTrg;
	this->vocab=vocab;

	string tag="decodings";
	Variable* tx=embedding.init("targetEmbedding", wordsTrg, hWordsTrg,
			vocabSize, embedSize, embedW);
	embeddingY=tx;

	Precision tDropout=params.dropout;
	XLLib::printfln(params.os,tag+".dropRt=%f", tDropout);
	int lstmInputSize=embedSize+embedSize;
	cell.lstmCell.init(tag+".lstmCell", batchSize, lstmInputSize, hiddenSize, numLayers_, 1,
			tDropout, hEncoder, cEncoder );

	tx=new Variable();  //Note memeory leak, but it feels safe for linCellAtt, linCellHaHt and linCellOut
	tx->resize(batchSize, hiddenSize, 1, 1);
	cell.linCellAtt.init(tag+".linCellAtt", tx, hiddenSize,false);

	tx=new Variable();
	tx->resize(batchSize, hiddenSize*2, 1, 1 );
	cell.linCellHaHt.init(tag+".linCellHaHt", tx, embedSize,false);

	tx=new Variable();
	tx->resize(batchSize, embedSize, 1, 1);
	cell.linCellOut.init(XLLib::stringFormat("%s.linCellOut", tag.c_str()),
			tx, vocabSize, true, 1, NULL, &(embedding.cell->w));

	for(int i=0;i<decodings.size();i++)
	{
		Variable* trgEmbedding=&trgEmbeddings[i];
		int tOffset=embedSize*batchSize*i;
		trgEmbedding->set(batchSize, embedSize, 1, 1, embeddingY->data+tOffset, embeddingY->grad.data+tOffset);
		Decoding* prev=i==0?NULL:&decodings[i-1];
		decodings[i].init(i, cell, trgEmbedding, hsSeq, prev, hiddenSize);
	}
	checkError(cudaDeviceSynchronize());
}


void Decodings::forward()
{
	int len=wordsTrg->nj;

	embedding.forward();

	for(int j=0;j<len-1;j++)
	{
		decodings[j].forward();
	}
}

Precision Decodings::getScore()
{
	int len=wordsTrg->nj;

	Precision likehood=0.0;
	Precision scale=1.0/batchSize;
	for(int i=0;i<len-1;i++)
	{
		Decoding& decoding=decodings[i];
		Precision tLikeli=decoding.softmax.backwardSmoothPrepare(wordsTrg->data+batchSize*(i+1), scale, true);
		likehood+=tLikeli;
	}

	return likehood;
}

void Decodings::backward()
{
	int len=wordsTrg->nj;

	{
		Decoding& decoding=decodings[len-2];
		decoding.lstm.setStateGradZero();
		decoding.dupHo.y1.grad.setZero();
	}

	DevMatInt trgWord;
	Precision likehood=0.0;
	embeddingY->grad.setZero();
	hsSeq->grad.setZero();
	Precision scale=1.0/batchSize;
	for(int i=len-2;i>=0;i--)
	{
		Decoding& decoding=decodings[i];
		decoding.backwardPrepare(wordsTrg, batchSize*(i+1),  scale);
		decoding.backward();
		likehood+=decoding.softmax.likehoodSum;
		cudaDeviceSynchronize();
	}

	embedding.backward();

	backwardScore=likehood;
	return;
}

void Decodings::calculateGradient()
{
	int len=wordsTrg->nj;
	for(int i=len-2;i>=0;i--)
	{
		decodings[i].calculateGradient();
	}

	embedding.calculateGradient();
}


void Decodings::getInitState(ModelState* state)
{
	LstmCell& lstmCell=cell.lstmCell;
	state->input.setZero();
	state->hx.copyFrom(*lstmCell.hx);
	state->cx.copyFrom(*lstmCell.cx);

}

void Decodings::transfer(ModelState* start, int word, ModelState* end, DevMatPrec& probs)
{
	embedding.apply(word);
	decodings[0].transfer(start, end, probs);

}

} /* namespace cytonLib */
