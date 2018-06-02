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

#include "StackState.h"

namespace cytonLib
{
void StackState::set(StackState* prev_, ModelState* modelState_, int word_, int word2_,
		Precision score_, Precision globalScore_)
{
	this->prev=prev_;
	this->modelState=modelState_;
	this->word=word_;
	this->word2=word2_;
	this->score=score_;
	this->globalScore=globalScore_;
}

} /* namespace cytonLib */
