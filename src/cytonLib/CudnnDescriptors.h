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

#ifndef _CYTONLIB_CUDNNDESCRIPTORS_H_
#define _CYTONLIB_CUDNNDESCRIPTORS_H_

#include "Variable.h"

namespace cytonLib
{

class CudnnDescriptors
{
public:

	int len;
	int maxLen;
	int n;
	int c;
	int h;
	int w;
	cudnnTensorDescriptor_t* descs;

	CudnnDescriptors();

	void init(int maxLen_, int n_, int c_, int h_=1, int w_=1);

	~CudnnDescriptors();

	static void createNdDesc(cudnnTensorDescriptor_t& desc, int d0, int d1, int d2);
};

} /* namespace cytonVR */

#endif /* CUDNNVARNCHW_H_ */
