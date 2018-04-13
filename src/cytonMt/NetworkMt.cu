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

#include "NetworkMt.h"
#include "WeightFactory.h"
#include "ParamsMt.h"

namespace cytonMt
{

void NetworkMt::init()
{
	weightFactory.init(params.optimization);
	batch.init(cytonLib::batchSize, params.maxSeqLen);
	decodings.preInit(params.maxSeqLen, params.mode);

	Variable* tx=embeddingSrc.init("sourceEmbedding", &batch.srcMat, &batch.hSrcMat,
			params.srcVocabSize, params.hiddenSize.at(0));
	layers.push_back(&embeddingSrc);

	tx=encoder.init( tx, params.maxSeqLen, batchSize, params.numLayers, params.hiddenSize.at(0));
	layers.push_back(&encoder);

	decodings.init( tx, encoder.hy, encoder.cy, &batch.trgMat, &batch.hTrgMat,
			params.trgVocabSize, params.hiddenSize.at(0), params.numLayers, &trgVocab);
	layers.push_back(&decodings);
	weightFactory.alloc(params.clipGradient);
}

void NetworkMt::backward()
{
	decodings.backward();
	encoder.backward();
	embeddingSrc.backward();
}


Precision NetworkMt::train(Precision lambda, bool updateParams)
{
	assert(cytonLib::testMode==false);

	this->forward();
	cudaDeviceSynchronize();

	this->backward();
	cudaDeviceSynchronize();
	Precision likehood=decodings.backwardScore;

	weightFactory.clearGrad();
	this->calculateGradient();

	if(updateParams)
	{
		weightFactory.update(lambda);
	}

	nBatches+=1;

	Precision likehood1=likehood*batch.factor;
	return likehood1;
}

Precision NetworkMt::getScore()
{

	this->forward();
	cudaDeviceSynchronize();

	Precision likehood=decodings.getScore();
	cudaDeviceSynchronize();

	return likehood;

}

void NetworkMt::saveModel(string modelName)
{
	weightFactory.save(modelName);
}

void NetworkMt::loadModel(string modelName)
{
	weightFactory.load(modelName);
}

void NetworkMt::prevApply()
{
}

string NetworkMt::apply(vector<string>& words, Precision& likeli)
{
	assert(testMode);

	embeddingSrc.forward();

	encoder.forward();

	return "";
}

void NetworkMt::getInitState(ModelState* state)
{
	decodings.getInitState(state);
}

void NetworkMt::prevBeamSearch()
{
	embeddingSrc.forward();
	encoder.forward();
}

void NetworkMt::beamSearchTransfer(ModelState* start, int input,
		ModelState* end, DevMatPrec& outputs)
{

	decodings.transfer(start, input, end, outputs);
}

}
