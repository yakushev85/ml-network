#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

#include "ml_network.h"

using namespace std;


struct ConfData
{
    string key;
    string value;
};

string vectorToString(vector<double>* x) 
{
	string result = "";

	for (double j : *x) 
	{
		result += to_string(int(round(j)));
	}

	return result;
}

ConfData getConfData(string rawline)
{
    ConfData resConfData;

    bool is_key = true;

    for (char c : rawline)
    {
        if (c == '=') 
        {
            is_key = false;
        } 
        else 
        {
            if (is_key)
            {
                resConfData.key += c;
            }
            else
            {
                resConfData.value += c;
            }
        }
    }

    return resConfData;
}

vector<int>* strToIntArray(string v)
{
    vector<int>* resArray = new vector<int>();
    string cValue = "";

    for (char c : v)
    {
        if (c == ' ') {
            if (cValue != "")
            {
                resArray->push_back(stoi(cValue));
                cValue = "";
            }
        }
        else
        {
            cValue += c;
        }
    }

    if (cValue != "") 
    {
        resArray->push_back(stoi(cValue));
    }

    return resArray;
}

vector<double>* strToDoubleArray(string v)
{
    vector<double>* resArray = new vector<double>;
    string cValue = "";

    for (char c : v)
    {
        if (c == ' ') {
            if (cValue != "")
            {
                resArray->push_back(stod(cValue));
                cValue = "";
            }
        }
        else
        {
            cValue += c;
        }
    }

    if (cValue != "") 
    {
        resArray->push_back(stod(cValue));
    }

    return resArray;
}

vector<TeachDataEntity>* loadData(string fn)
{
    vector<TeachDataEntity>* resultVector = new vector<TeachDataEntity>();
    ifstream dataFile;
    dataFile.open(fn, ios_base::in);

    if (dataFile.is_open()) 
    {
        string currentLine;
        bool is_inp = true;
        TeachDataEntity cTeachEntity;

        while (dataFile.good())
        {
            getline(dataFile, currentLine);

            if (is_inp) 
            {
                cTeachEntity = TeachDataEntity();
                cTeachEntity.inp = strToDoubleArray(currentLine);
                is_inp = false;
            }
            else
            {
                cTeachEntity.output = strToDoubleArray(currentLine);
                is_inp = true;
                resultVector->push_back(cTeachEntity);

                cout << vectorToString(cTeachEntity.inp) << " -> " << vectorToString(cTeachEntity.output) << endl;
            }
        }

        dataFile.close();
    }

    return resultVector;
}

NetConfiguration* loadConf(string configFn, string dataFn)
{
    NetConfiguration* nConfig = new NetConfiguration();

    ifstream confFile;
    confFile.open(configFn, ios_base::in);

    if (confFile.is_open()) 
    {
        string currentLine;

        while (confFile.good())
        {
            getline(confFile, currentLine);

            ConfData confData = getConfData(currentLine);

            if (confData.key == "inCount")
            {
                nConfig->inCount = stoi(confData.value);
                cout << "nConfig.inCount = " << nConfig->inCount << endl;
            }
            else if (confData.key == "neuronCounts")
            {
                nConfig->neuronCounts = strToIntArray(confData.value);
                cout << "nConfig.neuronCounts = ";

                for (int nCount : *(nConfig->neuronCounts))
                {
                    cout << nCount << " ";
                }

                cout << endl;
            }
            else if (confData.key == "maxLearningIterations")
            {
                nConfig->maxLearningIterations = stoi(confData.value);
                cout << "nConfig.maxLearningIterations = " << nConfig->maxLearningIterations << endl;
            }
            else if (confData.key == "initialWeightValue")
            {
                nConfig->initialWeightValue = stod(confData.value);
                cout << "nConfig.initialWeightValue = " << nConfig->initialWeightValue << endl;
            }
            else if (confData.key == "alpha")
            {
                nConfig->alpha = stod(confData.value);
                cout << "nConfig.alpha = " << nConfig->alpha << endl;
            }
            else if (confData.key == "speed")
            {
                nConfig->speed = stod(confData.value);
                cout << "nConfig.speed = " << nConfig->speed << endl;
            }
        }

        confFile.close();
    }

    nConfig->teachData = loadData(dataFn);
    cout << "nConfig.teachData.size = " << nConfig->teachData->size() << endl;

    return nConfig;
}

int main()
{
    string confFilename("./conf.props");
    string dataFilename("./data.txt");
    string examFilename("./exam.txt");
	cout << "Network initialization for ..." << endl;

	NetConfiguration* configuration = loadConf(confFilename, dataFilename);

	MultiNetwork* multiNetwork = new MultiNetwork(configuration);

    vector<TeachDataEntity>* examData = loadData(examFilename);
    cout << "examData.size = " << examData->size() << endl;

	cout << "Learning..." << endl;
	multiNetwork->learn(true);

	cout << "Testing..." << endl;
	for (TeachDataEntity testItem : *(configuration->teachData)) 
	{
		string outputStr = vectorToString(multiNetwork->execute(testItem.inp));
		string answerStr = vectorToString(testItem.output);
		string mark = (outputStr == answerStr)? " PASSED": " FAILED";

		cout << "Actual: " << outputStr << " Expected: " << answerStr << endl;
		cout << mark << endl;
	}

    cout << "Examing..." << endl;
    for (TeachDataEntity examItem : *examData) 
	{
		string outputStr = vectorToString(multiNetwork->execute(examItem.inp));
		string answerStr = vectorToString(examItem.output);
		string mark = (outputStr == answerStr)? " PASSED": " FAILED";

		cout << "Actual: " << outputStr << " Expected: " << answerStr << endl;
		cout << mark << endl;
	}

	cout << "Done." << endl;

    delete examData;
    delete multiNetwork;
    delete configuration;

	return 0;
}


	