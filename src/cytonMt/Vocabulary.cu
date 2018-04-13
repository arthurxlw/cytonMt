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

#include "Vocabulary.h"

namespace cytonMt {

int vocabEmpty=0;
int vocabSos=1;
int vocabEos=2;
int vocabUnk=3;

void Vocabulary::load(const string& file_name, int vocabSize)
{
	vector<string> ts;
	XLLib::readFile(file_name, ts);

	clear();
	empty=add("<EMPTY>");
	assert(empty==0);

	sos=add("<SOS>");
	eos=add("<EOS>");
	unk=add("<UNK>");
	for(vector<string>::iterator it=ts.begin();it!=ts.end();it++)
	{
		string t=*it;
		int i=t.find(' ');
		if(i>=0)
		{
			t=t.substr(0,i);
		}
		add(t);

		if(vocabSize>0 && size()>=vocabSize)
		{
			break;
		}
	}
}

void Vocabulary::save(string fileName)
{
	std::ofstream f(fileName.c_str());
	for(int i=4;i<texts.size();i++)
	{
		f<<texts[i]<<" "<<i<<"\n";
	}
	f.close();
}

void Vocabulary::clear()
{
	texts.clear();
	unordered_map<string,int>::clear();
}

void Vocabulary::parse(const string& line, vector<int>& res, bool ignoreUnk)
{
	vector<string> words;
	XLLib::str2list(line,words);

	int num_unk0=0;
	res.clear();
	for(vector<string>::iterator iw=words.begin(); iw!=words.end(); iw++)
	{
		string& word=*iw;
		int id=getId(word);
		if(id!=unk)
		{
			res.push_back(id);
		}
		else
		{
			num_unk0+=1;
			if(!ignoreUnk && (res.empty() || res.back()!=unk))
			{
				res.push_back(id);
			}
		}
	}
	return;
}

int Vocabulary::add(const string& word)
{
	int id=texts.size();
	texts.push_back(word);
	(*this)[word]=id;
	return id;
}

const string& Vocabulary::getText(int id)
{
	return texts.at(id);
}

string Vocabulary::getText(int id, int wordCase)
{
	string str=texts.at(id);
	if(wordCase==2)
	{
		int i=1;
		if(str.length()>=2 && str.at(0)=='_')
		{
			i=2;
		}
		std::transform(str.begin(), str.begin()+i,str.begin(), ::toupper);
	}
	else if(wordCase==3)
	{
		std::transform(str.begin(), str.end(),str.begin(), ::toupper);
	}
	return str;
}


string Vocabulary::getText(vector<int>& ids)
{
	std::ostringstream os;
	bool first=true;
	for(vector<int>::iterator it=ids.begin();it!=ids.end();it++)
	{
		if(!first)
		{
			os<<" ";
		}
		first=false;
		os<<texts.at(*it);
	}
	return os.str();

}

int Vocabulary::getId(const string& word)
{
	unordered_map<string,int>::iterator it=unordered_map<string,int>::find(word);
	if(it==end())
	{
		return unk;
	}
	else
	{
		return it->second;
	}
}

int Vocabulary::getCase(const string& word)
{
	int res=1;
	if(word.length()>=1)
	{
		char c=word.at(0);
		if(c>='A' && c<='Z')
		{
			res=2;
			if(word.length()>=2)
			{
				c=word.at(1);
				if(c>='A' && c<='Z')
				{
					res=3;
				}
			}
		}
	}
	return res;
}

} /* namespace cudaRnnTrans */
