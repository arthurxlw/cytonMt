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

#include "EmbeddingLayer.h"
#include <map>
#include "HostMatReal.h"

namespace cytonLib {


Variable* EmbeddingLayer::init(string tag, DevMatInt* x, HostMatInt* hx,
		int vocabSize, int hiddenSize, Weight* w)
{
	embeddingCell.init(tag, vocabSize, hiddenSize, w);
	Variable* res=EmbeddingInstance::init(tag, &embeddingCell, x, hx);
	return res;
}

Variable* EmbeddingLayer::init(string tag, DevMatInt* x, HostMatInt* hx,
		int vocabSize, int hiddenSize, Precision* y_, Precision* dy_, int stride)
{
	embeddingCell.init(tag, vocabSize, hiddenSize);
	Variable* res=EmbeddingInstance::init(tag, &embeddingCell, x, hx, y_, dy_, stride);
	return res;
}

} /* namespace cytonLib */
