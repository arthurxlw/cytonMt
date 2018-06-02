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

#include "Encoder.h"
#include "ParamsMt.h"

namespace cytonMt
{

Variable* Encoder::init(Variable* x, int maxSeqLen, int batchSize, int numLayers,
		int embedSize, int hiddenSize)
{
	if(hiddenSize%2!=0)
	{
		XLLib::printfln("hiddenSize must be divided by 2 for bidirectional RNN.");
		assert(false);
	}

	string tag="encoder";

	Precision tDropout=params.dropout;
	XLLib::printfln(params.os, "encoder.dropRs=%f", tDropout);
	Variable* tx=lstm.init(tag+".lstm", x, true, hiddenSize/2, numLayers, tDropout);
	layers.push_back(&lstm);

	hyMono.init(tag+".hyMono", &lstm.hy);
	layers.push_back(&hyMono);

	cyMono.init(tag+".cyConvert", &lstm.cy);
	layers.push_back(&cyMono);

	tx=transY.init(tag+".transY", tx);
	layers.push_back(&transY);

	y=tx;
	hy=&hyMono.y;
	cy=&cyMono.y;

	return tx;
}

} /* namespace cytonLib */
