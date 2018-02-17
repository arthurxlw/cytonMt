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

#include "xlCLib.h"
#include "ParamsMt.h"
#include "MachTrans.h"
#include "utils.h"

const char* version="2018-0210-0910";

using namespace cytonMt;
using namespace cytonLib;

int main(int argc, char** argv)
{
	XLLib::printfln("version: %s",version);

	XLLibTime start_time=XLLib::startTime();
	MachTrans machTrans;

	params.parse(argc, argv);

	cytonLib::global.init();

	machTrans.work();

	cytonLib::global.end();
	XLLib::printfln(params.os, "\n");
	XLLib::end(start_time);
	exit(0);

}



