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

#ifndef _CYTONMT_ATTENTION_H_
#define _CYTONMT_ATTENTION_H_

#include "basicHeadsMt.h"
#include "DuplicateLayer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"
#include "Variable.h"
#include "WeightedHs.h"
#include "MultiplyHsHt.h"
#include "Concatenate.h"
#include "ActivationLayer.h"
#include "Network.h"

using namespace cytonLib;

namespace cytonMt
{

class Attention: public Network
{
public:
	 DuplicateLayer dupHt;    // declare components
	 LinearLayer linearHt;
	 MultiplyHsHt multiplyHsHt;
	 SoftmaxLayer softmax;
	 WeightedHs weightedHs;
	 Concatenate concateCsHt;
	 LinearLayer linearCst;
	 ActivationLayer actCst;

	Variable* init(string tag, LinearLayer* linHt,
		    LinearLayer* linCst, Variable* hs, Variable* ht);
};



} /* namespace cytonLib */

#endif /* ATTENTION_H_ */
