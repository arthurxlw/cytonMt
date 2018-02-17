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

#ifndef _CYTONMT_EMBEDDINGINSTANCE_H_
#define _CYTONMT_EMBEDDINGINSTANCE_H_

#include "HostMatrix.h"
#include "DeviceMatrix.h"
#include "EmbeddingCell.h"
#include "Variable.h"
#include "Layer.h"

namespace cytonLib
{

class EmbeddingInstance: public Layer
{
public:

	EmbeddingCell* cell;

	DevMatInt* x;
	HostMatInt* hx;
	Variable y;

	DevMatInt firstOccurs;
	HostMatInt hFirstOccurs;

	DevMatInt deviceW;
	bool yResize;
	int stride;

public:
	Variable* init(string tag_, EmbeddingCell* cell, DevMatInt* x_, HostMatInt* hx_);

	void forward();

	void backward();

	void calculateGradient();

	void apply(int word);

	static void makeFirstOccurs(int* words, int* firstOccurs, int n);

	Variable* init(string tag_, EmbeddingCell* cell_,
			DevMatInt* x_, HostMatInt* hx_, Precision* y_, Precision* dy_, int stride);

};

} /* namespace cytonLib */

#endif /* EMBEDDINGLAYER_H_ */
