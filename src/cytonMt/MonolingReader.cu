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

#include "MonolingReader.h"
#include "CorpusReader.h"

namespace cytonMt
{

void MonolingReader::init(const string& file_name_, Vocabulary* vocab_)
{
	fileName=file_name_;
	vocab = vocab_;
}

void MonolingReader::reset()
{
	if(fileName!="stdin")
	{
		file.close();
		file.open(fileName);
	}

}

void MonolingReader::closeFile()
{
	file.close();
}


int MonolingReader::read_mini_batch(int batchSize, bool ignoreUnk, HostMatInt& matrix, string* raw)
{
	vector<vector<int>> sents;
	int maxLen=0;
	vector<int> sent;
	for(int i=0;i<batchSize;i++)
	{
		string line;
		bool read=true;
		if(fileName!="stdin")
		{
			read=std::getline(file, line, '\n');
		}
		else
		{
			read=std::getline(std::cin, line, '\n');
		}

		if(read && !line.empty())
		{
			vocab->parse(line, sent, ignoreUnk);
			sents.push_back(sent);
			maxLen=std::max(maxLen, (int)sent.size());

			*raw += line+"\n";
		}
		if(!read)
		{
			break;
		}
	}

	if(!sents.empty())
	{
		packMatrix(sents, maxLen, batchSize, matrix, true, vocab->sos, vocab->eos, vocab->empty);
	}
	return sents.size();
}



} /* namespace cudaRnnTrans */
