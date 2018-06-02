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

#ifndef _CYTONMT_EMBEDDINGCELL_H_
#define _CYTONMT_EMBEDDINGCELL_H_

#include "DeviceMatrix.h"
#include "HostMatReal.h"
#include "DevMatReal.h"
#include "Weight.h"

namespace cytonLib
{

class EmbeddingCell
{
public:
	int vocabSize;
	int hiddenSize;
	Weight w;

public:
	void init(string tag, int vocabSize, int hiddenSize, Weight* w0=NULL);
};

} /* namespace cytonLib */

#endif /* EMBEDDINGLAYER_H_ */
