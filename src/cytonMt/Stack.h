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

#ifndef _CYTONMT_STACK_H_
#define _CYTONMT_STACK_H_

#include "basicHeads.h"
#include "StackCell.h"
#include "Vocabulary.h"

namespace cytonLib
{

class Stack: public vector<StackCell>
{
public:
	void init(int maxLen, int beamSize);



};

} /* namespace cytonLib */

#endif /* STACK_H_ */
