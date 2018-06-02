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

#include "BeamSearch.h"

namespace cytonLib
{

inline void findMaxElements(Precision* data, int len,
		int num, vector<int>& idxs, vector<Precision>& values)
{
	idxs.clear();
	values.clear();
	thrust::device_ptr<Precision> dev_ptr = thrust::device_pointer_cast(data);
	for(int k=0;k<min(num, len); k++)
	{
		thrust::device_ptr<Precision> iter =
						thrust::max_element(dev_ptr, dev_ptr+len);
		int idx=&iter[0]-&dev_ptr[0];
		idxs.push_back(idx);
		values.push_back(*iter);
		*iter=-1e6;
	}
}

Precision fGlobalScore(Precision score, int length, Precision lenPenalty)
{
	Precision factor=pow((5.0+1.0)/(5.0+(Precision)length), lenPenalty);
	Precision res=score*factor;
	return res;
}

void BeamSearch::init(cytonMt::NetworkMt* model_, cytonMt::Vocabulary* vocab_,
		int beamSize_, int maxSeqLen_, Precision lenPenalty_,
		int embedSize, vector<int>& hiddenSize, int numLayers)
{
	model=model_;
	vocab=vocab_;
	beamSize=beamSize_;
	maxSeqLen1=maxSeqLen_+1;
	lenPenalty=lenPenalty_;

	initModelState.init(embedSize, hiddenSize.at(0), numLayers);
	initStackState.set(NULL, &initModelState, vocab->sos, vocab->empty, 0, 0);

	stack.init(maxSeqLen1, beamSize);
	stack[0].push_back(&initStackState);

	int maxNumStates=beamSize*beamSize*maxSeqLen1+5;
	modelStateFactory.init(maxNumStates);
	for(vector<ModelState>::iterator it=modelStateFactory.vec.begin();
			it!=modelStateFactory.vec.end(); it++)
	{
		it->init(embedSize, hiddenSize.at(0), numLayers);
	}

	stackStateFactory.init(maxNumStates);
}

string BeamSearch::apply(vector<string>& trans, Precision* resScore)
{
	model->prevBeamSearch();
	modelStateFactory.reset();
	stackStateFactory.reset();

	model->getInitState(&initModelState);
	ModelState* nextModelState=modelStateFactory.get();

	vector<int> idxs;
	vector<Precision> values;
	int iStack=0;
	for(;iStack<maxSeqLen1-1; iStack++)
	{
		StackCell& curStack=stack[iStack];
		bool finished=true;
		for(StackCell::iterator it=curStack.begin(); it!=curStack.end(); it++)
		{
			if( (*it)->word!=vocab->eos)
			{
				finished=false;
				break;
			}
		}
		if(finished)
		{
			break;
		}

		StackCell& nextStack=stack[iStack+1];
		nextStack.clear();
		for(int iState=0;iState<curStack.size();iState++)
		{
			StackState* curState=curStack[iState];
			if(curState->word==vocab->eos)
			{
				if(nextStack.accept(curState->globalScore))
				{
					nextStack.put(curState);
				}
			}
			else
			{
				model->beamSearchTransfer(curState->modelState, curState->word, nextModelState, probs);

				findMaxElements(probs.data, probs.length(), beamSize, idxs, values);

				bool accepted=false;
				for(int k=0;k<idxs.size();k++)
				{
					int word=idxs[k];
					Precision value=values[k];
					Precision nextScore=curState->score+log(value);
					Precision globalScore=fGlobalScore(nextScore, iStack+1, lenPenalty);
					if(nextStack.accept(globalScore))
					{
						StackState* nextState=stackStateFactory.get();
						nextState->set(curState, nextModelState, word, 0, nextScore, globalScore);
						nextStack.put(nextState);
						accepted=true;
					}
				}
				if(accepted)
				{
					nextModelState=modelStateFactory.get();
				}
			}
		}
	}

	StackState* bestEndState=stack.at(iStack).at(0);
	string detail = getDerivation(bestEndState, trans);
	if(resScore!=NULL)
	{
		*resScore=bestEndState->score;
	}
	return detail;
}

string BeamSearch::getDerivation(StackState* lastState, vector<string>& words)
{
	vector<StackState*> states;
	StackState* state=lastState;
	while(state!=NULL)
	{
		states.push_back(state);
		state=state->prev;
	}

	words.clear();
	ostringstream os;
	os<<"";
	bool first=true;
	for(int i=states.size()-2; i>=0; i--)
	{
		state=states[i];
		string word=vocab->getText(state->word, state->word2);
		Precision logProb=state->score-states[i+1]->score;

		if(!first)
		{
			os<<" ";
		}
		first=false;

		os<<XLLib::stringFormat("%s|%e", word.c_str(), exp(logProb));

		if(i!=0)
		{
			words.push_back(word);
		}

	}
	string detail=os.str();
	return detail;

}

} /* namespace cytonLib */
