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

#include "ParamsMt.h"
#include "Global.h"

namespace cytonMt
{

ParamsMt::ParamsMt()
{
	const Option options[] = {
			{"mode", "", "train/translate"},
			{"saveModel", "", ""},
			{"loadModel", "", "load model for continue training or translate"},
			{"maxSaveModels", "10","maximum number of saved models"},
			{"train", "trainSrc:trainTrg", "source-side and target-side training files, one sentences per line"},
			{"dev",  "devSrc:devTrg",	"source-side and target-side development files, one sentences per line"},
			{"testInput", "testInput", "input file for translating"},
			{"testOutput", "testOutput", "output file for translating"},
			{"vocab",  "vocabSrc:vocabTrg",	"source-side and target-side vocabulary files, one word per line"},
			{"srcVocabSize",  "0",	"size of source-side vocabulary, 0 means using whole vocabulary in vocabSrc file"},
			{"trgVocabSize",  "0",	"size of source-side vocabulary, 0 means using whole vocabulary in vocabTrg file"},
			{"ignoreUnk",  "1",	"0/1, 1 means ignoring unknown words"},

			{"initParam",  "0.1",	"initialize weights uniformly in (-initParam, initParam)"},
			{"optimization",  "SGD",	"SGD/Adam"},
			{"learningRate",  "1",	"learning rate"},
			{"decayRate",  "0.7",	"decay factor of learning rate"},
			{"decayStart",  "1000",	"learning rate start to decay from the epoch of decayStart"},
			{"decayConti", "0", "0/1, 1 means that learning rate keeps decaying per check once it decays, OpenNMT's mode,  "},
			{"decayStatus", "0", "0/1, 1 means that learning rate is in a status of decaying, useful for continue training."},

			{"epochs",  "100",	"max epochs of training"},
			{"epochStart", "1", "the number of first epoch, useful for continue training"},
			{"batchSize",  "64",	"batch size"},
			{"maxSeqLen",  "100",	"max length of source and target sentence"},
			{"hiddenSize",  "512",	"size of word embedding and hidden states"},
			{"numLayers",  "2",	"number of encoder/decoder layers"},
			{"dropout",  "0.2",	"dropout rate, 0 means disabling dropout"},
			{"clipGradient",  "5",	"threshold for clip gradient"},
			{"labelSmooth", "0.1", "factor of smoothing the target labels"},
			{"probeFreq", "1", "number of times probing the development likelihood per epoch"},
			{"probeMargin", "0.01", "margin for checking whether the development likelihood has increased"},
			{"patience", "1", "threshold for decaying the learning rate and restart training from the best model"},

			{"beamSize", "10", "size of beam search in translating"},
			{"lenPenalty", "0.6", "length penalty"},
			{"","",""}
	};

	addOptions(options);
}

void ParamsMt::init_members()
{
	mode=opt2val["--mode"];
	saveModel=get("saveModel");
	if(!saveModel.empty())
	{
		saveModel+="/";
	}
	loadModel=get("loadModel");
	maxSaveModels=geti("maxSaveModels");
	trainData=get("train");
	devData=get("dev");
	testInput=get("testInput");
	testOutput=get("testOutput");
	vector<string> ts;
	XLLib::str2list(get("vocab"),":", ts);
	if(!ts.empty())
	{
		if(ts.size()!=2)
		{
			XLLib::printfln("the parameter of vocab is wrong: %s", get("vocab"));
			exit(1);
		}
		srcVocab=ts.at(0);
		trgVocab=ts.at(1);
	}
	srcVocabSize=geti("srcVocabSize");
	trgVocabSize=geti("trgVocabSize");
	ignoreUnk=geti("ignoreUnk");

	initParam=getf("initParam");
	optimization=get("optimization");
	learningRate=getf("learningRate");
	decayRate=getf("decayRate");
	decayStart=getf("decayStart");
	decayConti=geti("decayConti");
	decayStatus=geti("decayStatus");

	epochs=geti("epochs");
	epochStart=geti("epochStart");
	cytonLib::batchSize=geti("batchSize");
	maxSeqLen=geti("maxSeqLen");
	XLLib::str2ints(get("hiddenSize"), ":", hiddenSize);
	numLayers=geti("numLayers");
	dropout=getf("dropout");
	clipGradient=getf("clipGradient");
	labelSmooth=getf("labelSmooth");
	probeFreq=getf("probeFreq");
	probeMargin=getf("probeMargin");
	patience=geti("patience");

	beamSize=geti("beamSize");
	lenPenalty=getf("lenPenalty");

	if(!loadModel.empty())
	{
		XLLib::str2list(loadModel, ":", ts);
		string tModel=ts.at(0);
		int i=tModel.rfind("/");
		string tDir=tModel.substr(0,i+1);
		string tFile=tDir+"/settings";
		XLLib::printfln(os, "load arguments from %s", tFile.c_str());
		loadModelParams(tFile);
	}
}

void ParamsMt::saveModelParams(std::string fileName)
{
	std::ofstream f(fileName.c_str());
	f<<numLayers<<"\n";
	f<<XLLib::toString_vec_ostream(hiddenSize,":")<<"\n";
	f<<srcVocabSize<<"\n";
	f<<trgVocabSize<<"\n";
	f<<cytonLib::batchSize<<"\n";
	f<<maxSeqLen<<"\n";
	f.close();
}

void ParamsMt::loadModelParams(std::string fileName)
{
	std::ifstream f(fileName.c_str());
	string t;
	f>>numLayers;

	getline(f, t);
	if(t.empty())
	{
		getline(f, t);
	}
	hiddenSize.clear();
	XLLib::str2ints(t, ":", hiddenSize);

	getline(f, t);
	srcVocabSize=atoi(t.c_str());

	getline(f, t);
	trgVocabSize=atoi(t.c_str());

	f>>cytonLib::batchSize;

	int tn;
	f>>tn;
	maxSeqLen=std::max(maxSeqLen, tn);

	f.close();
}

ParamsMt params;

}
