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

#ifndef _CYTONMT_DECODINGCELL_H_
#define _CYTONMT_DECODINGCELL_H_

#include "basicHeads.h"
#include "LstmCell.h"
#include "LinearLayer.h"
#include "LstmInstance.h"


using namespace cytonLib;

namespace cytonMt
{

class DecodingCell
{
public:
		LstmCell lstmCell;
		LinearLayer linCellAtt;
		LinearLayer linCellHaHt;
		LinearLayer linCellOut;
};

} /* namespace cytonLib */

#endif /* DECODINGCELL_H_ */
