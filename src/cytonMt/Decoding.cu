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

#include "Decoding.h"
#include <thrust/extrema.h>
#include "ParamsMt.h"

namespace cytonMt
{
extern ParamsMt params;

inline int findMaxElement(Precision* data, int len, Precision* value)
{
	thrust::device_ptr<Precision> dev_ptr = thrust::device_pointer_cast(data);

	thrust::device_ptr<Precision> iter =
			thrust::max_element(dev_ptr, dev_ptr+len);
	int index=&iter[0]-&dev_ptr[0];
	*value=iter[0];
	return index;
}

void Decoding::init(int index, DecodingCell& cell, Variable* embeddingY, Variable* hs,
		Decoding* prev, int hiddenSize)
{
	this->index=index;
	this->embeddingY=embeddingY;

	string tag=XLLib::stringFormat("decoding%d", index);

	Variable* ho=NULL;
	if(prev!=NULL)
	{
		ho=& prev->dupHo.y1;
	}
	else
	{
		ho=new Variable();
		ho->resize(batchSize, hiddenSize, 1, 1);
		ho->setZero();
		ho->frozen=true;
	}

	Variable* tx=NULL;
	tx=concateOutEmbed.init(tag+".concateOutEmbed", ho, embeddingY);
	layers.push_back(&concateOutEmbed);

	LstmCell* tCell=&cell.lstmCell;
	if(prev!=NULL)
	{
		LstmInstance* prevLstm=&prev->lstm;
		tx=lstm.init(tag+".lstm", tCell, tx, &prevLstm->hy, &prevLstm->cy);
	}
	else
	{
		tx=lstm.init(tag+".lstm", tCell, tx, tCell->hx, tCell->cx);
	}
	layers.push_back(&lstm);

	tx=attention.init(tag+XLLib::stringFormat(".att%d", index),
			&cell.linCellAtt, &cell.linCellHaHt, hs, tx);
	layers.push_back(&attention);

	Precision tDropout=params.dropout;
	if(index==0)
	{
		XLLib::printfln(global.os,"decoding.dropHo=%f", tDropout);
	}
	tx=dropHo.init(tag+".dropOut", tx, tDropout);
	tx->frozen=true;
	layers.push_back(&dropHo);

	tx=dupHo.init(tag+".duplicator", tx);
	layers.push_back(&dupHo);

	tx=linearHo.init(tag+".linOut", &cell.linCellOut, tx);
	layers.push_back(&linearHo);

	softmax.init(tag+".softmax", tx);
	layers.push_back(&softmax);
}

void Decoding::transfer(ModelState* start, ModelState* end, DevMatPrec& probs)
{
	concateOutEmbed.x0->setForced(start->input.nj, start->input.ni, 1, 1,
			start->input.data, NULL);

	lstm.prevTransfer(&start->hx, &start->cx, &end->hx, &end->cx);

	this->forward();

	end->input.copyFrom(dropHo.y);

	probs.set(softmax.y.ni, softmax.y.ni, softmax.y.nj, softmax.y.data);
}

void Decoding::backwardPrepare(DevMatInt* wordTrg, int offset,  Precision scale)
{
	softmax.backwardSmoothPrepare(wordTrg->data+offset, scale, true, params.labelSmooth);
}

} /* namespace cytonLib */
