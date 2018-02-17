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

#include "Attention.h"
#include "Global.h"

namespace cytonMt
{

Variable* Attention::init(string tag, LinearLayer* linHt,
    LinearLayer* linCst, Variable* hs, Variable* ht)
 {
  Variable* tx;
  tx=dupHt.init(tag, ht);          // make two copies
  layers.push_back(&dupHt);

  tx=linearHt.init(tag, linHt, tx);           // WsHt
  layers.push_back(&linearHt);

  tx=multiplyHsHt.init(tag, hs, tx);        // HsWsHt
  layers.push_back(&multiplyHsHt);

  tx=softmax.init(tag, tx);                  // F_att
  layers.push_back(&softmax);

  tx=weightedHs.init(tag, hs, tx);              // Cs
  layers.push_back(&weightedHs);

  tx=concateCsHt.init(tag, tx, &dupHt.y1);      // Cst
  layers.push_back(&concateCsHt);

  tx=linearCst.init(tag, linCst, tx);
  layers.push_back(&linearCst);

  tx=actCst.init(tag, tx, CUDNN_ACTIVATION_TANH);// Ho
  layers.push_back(&actCst);

  return tx; //pointer to result
 }


} /* namespace cytonMt */
