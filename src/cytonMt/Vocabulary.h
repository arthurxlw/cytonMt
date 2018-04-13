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

#ifndef _CYTONMT_VOCABULARY_H_
#define _CYTONMT_VOCABULARY_H_

#include "basicHeadsMt.h"

namespace cytonMt {

extern int vocabEmpty;
extern int vocabSos;
extern int vocabEos;
extern int vocabUnk;

class Vocabulary : public unordered_map<string,int>
{

public:
	vector<string> texts;
	int empty;
	int sos;
	int eos;
	int unk;

	void load(const string& file_name, int vocabSize);

	void clear();

	void parse(const string& line, vector<int>& res, bool ignoreUnk);

	int add(const string& word);

	const string& getText(int id);

	string getText(int id, int wordCase);

	string getText(vector<int>& ids);

	int getId(const string& word);

	int getCase(const string& word);

	inline int size()
	{
		return texts.size();
	}

	void save(string fileName);
};

} /* namespace cudaRnnTrans */

#endif /* VOCABULARY_H_ */
