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

#ifndef _CYTONMT_STACKCELL_H_
#define _CYTONMT_STACKCELL_H_

#include "basicHeads.h"
#include "StackState.h"
#include "ModelState.h"

namespace cytonLib
{

class StackCell: public vector<StackState*>
{
	int beamSize;
public:

	void init(int beamSize_)
	{
		beamSize=beamSize_;
	}

	bool accept(Precision score);

	void put(StackState* state);

};

} /* namespace cytonLib */

#endif /* STACKCELL_H_ */
