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
#include "MachTrans.h"
#include "ParamsMt.h"
#include "Global.h"
#include "utils.h"
#include "WeightFactory.h"
#include "HostMatrix.h"
#include "CorpusReader.h"

#include "BeamSearch.h"
#include "MonolingReader.h"
#include "cublasWrapper.h"

using namespace cytonLib;

namespace cytonMt
{

bool MachTrans::checkTrainStatus(bool tune)
{
	double timeCost;
	XLLib::endTime(lastCheckTime, &timeCost);
	double likeli=sumTrainLikeli/nTrgWords;

	XLLib::printf(" s%.1e %s %0.0fw/s, lr:%.2e tr:%.3e",
			(double)nSents, checkTime().c_str(), nSrcWords/timeCost, lambda, likeli);

	XLLib::printf(" bestV:%.6f %.6f", likeliValidBest0, likeliValidBest1);
	Precision likeliValid=test(validCorpora);
	if(tune)
	{
		lastProbe=nSents;
		Precision margin=std::max((Precision)params.probeMargin*lambda, (Precision)0.001);
		bool succeed=likeliValid>=likeliValidBest0+margin;
		bool succeedRelax=likeliValid>=likeliValidBest1;

		XLLib::printf(" inc:%.6f/%.6f %s%s",  likeliValid-likeliValidBest0,
				margin, succeed?"s":"f", succeedRelax?"s":"f" );

		string tModel=params.saveModel+XLLib::stringFormat("/model_epoch%02d_s%d_%s%.6f",
				epoch, global.batch, succeedRelax?"s":"f", likeliValid);
		while(tModel==bestModel)
		{
			tModel+="a";
		}

		saveModel(tModel);
		XLLib::printf( " save:%s", XLLib::fileName(tModel).c_str());
		savedModels.push_back(tModel);

		if(succeed)
		{
			numFails=0;
			likeliValidBest0=std::max(likeliValid, likeliValidBest0);
		}
		else
		{
			numFails+=1;
		}
		XLLib::printf(" nFail:%d", numFails);
		if(numFails>=params.patience || epoch+1>=params.decayStart) //prepare for restart
		{
			lambdaReduced=true;
			likeliValidBest0=std::max(likeliValidBest0, likeliValidBest1);
		}

		if(succeedRelax)
		{
			string oldBest=bestModel;
			bestModel=tModel;
			XLLib::fileLink(XLLib::fileName(tModel), XLLib::stringFormat("%s/model", params.saveModel.c_str()));
			likeliValidBest1=std::max(likeliValid, likeliValidBest1);
		}
		else
		{
			if(numFails>=params.patience && bestModel!=tModel)
			{
				XLLib::printf(" loadM:%s", XLLib::fileName(bestModel).c_str());
				NetworkMt::loadModel(bestModel);
				test(validCorpora);
			}
		}


		if(lambdaReduced)
		{
			XLLib::printf(" lrDecay");
			lambda *= params.decayRate;
			numFails=0;

			if(!params.decayConti)
			{
				lambdaReduced=false;
			}
		}

		while(savedModels.size()>params.maxSaveModels)
		{
			for(int i=0; i<savedModels.size();i++)
			{
				string tm=savedModels.at(i);
				if(tm!=bestModel)
				{
					XLLib::fileRemove(tm);
					savedModels.erase(savedModels.begin()+i);
					break;
				}
			}
		}

		lastCheckTime=XLLib::startTime();
		sumTrainLikeli=0;
		nSrcWords=0;
		nTrgWords=0;
	}
	XLLib::printfln("");
	return false;
}
void MachTrans::learn(CorpusReader& corpus, bool updateParams)
{
	XLLibTime epochStart=XLLib::startTime();
	corpus.reset();
	CorpusReadNode* node;
	int nPrintDetails=1;

	HostMatInt srcMat;
	HostMatInt trgMat;
	HostMatInt trgMatSoftmax;
	while(true)
	{
		bool read=corpus.read(node);

		bool probe=false;
		bool tune=true;
		probe=nSents-lastProbe>=probePeriod;

		if(!probe && epoch==params.epochStart && nPrintDetails<=3 && nSents-lastProbe >=batchSize*50*nPrintDetails)
		{
			nPrintDetails+=1;
			probe=true;
			tune=false;
		}

		if(probe)
		{
			bool exit=checkTrainStatus(tune);
			fflush(stdout);
		}

		if(!read)
		{
			break;
		}

		global.batch+=1;
		batch.setSrcTrg(node->srcMat, node->trgMat, node->factor);

		double tLikeli=train(lambda, updateParams);

		sumTrainLikeli+=tLikeli;
		nSents+=batchSize*node->factor;
		nSrcWords+=batch.hSrcMat.length()*node->factor;
		nTrgWords+=batch.numTrgWords()*node->factor;
	}

}


double MachTrans::test(vector<CorpusReader>& corpora)
{
	bool gbTestMode=true;
	std::swap(cytonLib::testMode, gbTestMode);

	double res=0;
	XLLib::printf(" valid:");

	for(int i=0;i<corpora.size();i++)
	{
		double score=test(corpora[i]);
		if(i==0)
		{
			res=score;
		}
		else
		{
			XLLib::printf(" ");
		}
		XLLib::printf("%.6f", score);

	}

	std::swap(cytonLib::testMode, gbTestMode);
	return res;

}
double MachTrans::test(CorpusReader& corpus)
{
	double timeStartEpoch=clock();
	corpus.reset();
	double sumLikeli=0;
	int numSents=0;
	int numSrcWords=0;
	int numTrgWords=0;
	CorpusReadNode* node;

	bool gbTestMode=true;
	std::swap(cytonLib::testMode,gbTestMode);

	while(corpus.read(node))
	{
		batch.setSrcTrg(node->srcMat, node->trgMat, node->factor);
		double tLikeli=getScore()*node->factor;

		sumLikeli+=tLikeli;
		numSents+=batchSize*node->factor;
		numSrcWords+=batch.srcMat.length()*node->factor;
		numTrgWords+=batch.numTrgWords()*node->factor;
	}
	double likeliTrain=sumLikeli/numTrgWords;
	std::swap(cytonLib::testMode,gbTestMode);

	return likeliTrain;
}


string MachTrans::checkTime()
{
	string res=XLLib::stringFormat("%s %s", XLLib::endTime(startTime).c_str(),
			XLLib::endTime(lastCheckTime).c_str());
	return res;
}

void MachTrans::work()
{
	startTime=XLLib::startTime();
	string mode=params.mode;

	if(mode=="train")
	{
		workTrain();
	}
	else if(mode=="translate")
	{
		workTest();
	}
	else
	{
		XLLib::printfln("Unknown mode %s.",mode.c_str());
	}
}

void MachTrans::loadModel(string& modelFile)
{
	int i=modelFile.rfind('/');
	if(i<0)
	{
		XLLib::printfln("Error: modelFile wrong %s", modelFile.c_str());
		XLLib::printfln("modelFile must be modelDir/modelFile, and modelDir has settings vocab.sn vocab.tn.");
	}
	assert(i>=0);
	string modelDir=modelFile.substr(0, i+1);
	srcVocab.load(modelDir+"vocab.sn", 0);
	trgVocab.load(modelDir+"vocab.tn", 0);
	assert(params.srcVocabSize==srcVocab.size());
	assert(params.trgVocabSize==trgVocab.size());

	NetworkMt::init();
	NetworkMt::loadModel(modelFile);

}

void MachTrans::workTrain()
{
	cytonLib::testMode=false;
	if(XLLib::dirExists(params.saveModel))
	{
		XLLib::printfln("Warning: model dir exists : %s . ", params.saveModel.c_str());
//		exit(1);
	}

	XLLib::dirMake(params.saveModel);
	if(params.loadModel.empty())
	{
		srcVocab.load(params.srcVocab, params.srcVocabSize);
		params.srcVocabSize=srcVocab.size();

		trgVocab.load(params.trgVocab, params.trgVocabSize);
		params.trgVocabSize=trgVocab.size();

		NetworkMt::init();
	}
	else
	{
		loadModel(params.loadModel);
	}

	params.saveModelParams(params.saveModel+"settings");
	srcVocab.save(params.saveModel+"vocab.sn");
	trgVocab.save(params.saveModel+"vocab.tn");
	XLLib::printfln("real vocabSize src %d, trg %d", srcVocab.size(), trgVocab.size());

	CorpusReader trainCorpus;
	{
		vector<string> ts;
		XLLib::str2list(params.trainData,":", ts);
		assert(ts.size()==2);
		trainCorpus.init(ts.at(0), ts.at(1), srcVocab, trgVocab,  params.ignoreUnk, batchSize, params.maxSeqLen);
	}

	{
		validCorpora.push_back(CorpusReader());
		vector<string> ts;
		XLLib::str2list(params.devData,":", ts);
		validCorpora.back().init(ts.at(0),ts.at(1), srcVocab, trgVocab, params.ignoreUnk, batchSize, params.maxSeqLen);
	}

	Precision likeliValid=-10000;
	{
		XLLib::printf("initial");
		likeliValid=test(validCorpora);
		XLLib::printfln("");
		fflush(stdout);
	}

	bestModel=params.saveModel+XLLib::stringFormat("/model_epoch%d_s0%.6f",params.epochStart, likeliValid);
	saveModel(bestModel);
	XLLib::printfln("save:%s", XLLib::fileName(bestModel).c_str());
	savedModels.push_back(bestModel);

	likeliValidBest0=likeliValid;
	likeliValidBest1=likeliValid;

	lastCheckTime=XLLib::startTime();
	numFails=0;
	global.batch=0;
	nSents=0;
	nSrcWords=0;
	nTrgWords=0;

	lastProbe=0;
	lambda=params.learningRate;
	lambdaReduced=params.decayStatus;
	probePeriod=trainCorpus.nSents/params.probeFreq;
	XLLib::printfln("probePeriod %d sents (%d/%g)", probePeriod, trainCorpus.nSents, params.probeFreq);
	for(epoch=params.epochStart; epoch<=params.epochs; epoch++)
	{
		global.epoch=epoch;
		XLLib::printf("# e%d", epoch);
		learn(trainCorpus, true);
	}
	if(nSents-lastProbe>0)
	{
		checkTrainStatus();
	}

	if(params.epochs>0){
		XLLib::printfln("\nbestModel %s", bestModel.c_str());
		NetworkMt::loadModel(bestModel);
		XLLib::printf("best ");
		double avgLikeliValid=test(validCorpora);
		XLLib::printf("\n  %.16e", avgLikeliValid);
	}
}

void MachTrans::workTest()
{
	cytonLib::batchSize=1;
	cytonLib::testMode=true;
	loadModel(params.loadModel);

	BeamSearch beamSearch;
	beamSearch.init(this, &trgVocab, params.beamSize, params.maxSeqLen, params.lenPenalty,
				params.hiddenSize, params.numLayers);

	MonolingReader corpus;
	corpus.init(params.testInput, &srcVocab);
	corpus.reset();
	ofstream fOutput;
	ostream *out;
	if(params.testOutput!="stdout")
	{
		fOutput.open(params.testOutput);
		out=&fOutput;
	}
	else
	{
		out=&std::cout;
	}

	int batchSize=1;
	int nIgnore=0;
	HostMatInt srcMat;

	prevApply();
	double sumScore;
	int numSents=0;
	int numTrgWords=0;
	cudaSetDevice(0);
	XLLib::printfln(global.os, "start translating..");
	for(int i=0;;i++)
	{
		cudaDeviceSynchronize();

		string srcLine;
		int nRead=corpus.read_mini_batch(batchSize, params.ignoreUnk, srcMat, &srcLine);
		if(nRead==0)
		{
			break;
		}

		vector<string> trans;
		string detail;
		Precision score;
		int seqLen=srcMat.nj;
		if(seqLen>params.maxSeqLen)
		{
			nIgnore+=1;
			XLLib::printfln("%d-th (total %d) sentence is %d words, too long, so ignored.",
					i, nIgnore, seqLen);
		}
		else
		{
			if(srcMat.length()>0)
			{
				batch.setSrc(srcMat);
				detail=beamSearch.apply(trans, &score);
			}
		}
		sumScore+=score;
		numSents+=1;
		int tLen=trans.size()+1;
		numTrgWords+=tLen;
		XLLib::printfln(global.os, "%d %s => %e %s", i, srcLine.c_str(),
				score/tLen, detail.c_str());

		(*out)<<XLLib::toString(trans)<<"\n";
	}
	if(params.testOutput!="stdout")
	{
		fOutput.close();
	}

	double avgScore=sumScore/numTrgWords;
	XLLib::printfln(global.os, "sentences %d, ignore %d, words %d, avgScore %e .", numSents, nIgnore, numTrgWords, avgScore);
}


}
