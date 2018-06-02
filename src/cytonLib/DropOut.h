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

#ifndef DROPOUT_H_
#define DROPOUT_H_

#include "Global.h"
#include "Variable.h"
#include "Layer.h"

namespace cytonLib
{

class DropOut: public Layer
//note: The implement of dropout in cudnn seems has a bug.
//      I have encountered memory errors from CuDropoutLayer for a few times.
//      Therefore, using this implement is safer.
{
public:
	Precision dropout;
	bool localTestMode;

	DevMatPrec mask;
	bool active;

	Variable* init(string tag_, Variable* x, Precision dropout);

	void forward();

	void backward();
};

} /* namespace cytonLib */

#endif /* DROPOUT_H_ */
