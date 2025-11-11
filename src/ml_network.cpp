#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include "ml_network.h"

using namespace std;

Neuron::Neuron(int inCount, double initialWeightValue) 
{
    this->inCount = inCount;
    this->weightOffset = 0.0;

    inVector = nullptr;
    weights = new vector<double>();
    delta = new vector<double>();

    for (int i=0;i<this->inCount;i++) 
    {
        this->weights->push_back(initialWeightValue * (0.1 + 0.8*drand48()));
        this->delta->push_back(0.0);
    }
}

Neuron::~Neuron()
{
    delete inVector;
    delete weights;
    delete delta;
}

double Neuron::generateOutput() 
{
    net = weightOffset;
    for (int i=0;i<inCount;i++) 
    {
        net += weights->at(i)*inVector->at(i);
    }

    output = 1 / (1 + exp(-1 * net));
    
    return output;
}

Layer::Layer(int inCount, int neuronLayerCount, double initialWeightValue) 
{
    neurons = new vector<Neuron*>();
    for (int i=0;i<neuronLayerCount;i++) 
    {
        neurons->push_back(new Neuron(inCount, initialWeightValue));
    }
}

Layer::~Layer()
{
    delete neurons;
}

vector<Neuron*>* Layer::getNeurons()
{
    return neurons;
}


MultiNetwork::MultiNetwork(NetConfiguration* configuration)
{
    this->configuration = configuration;

    vector<int>* neuronCounts = configuration->neuronCounts;
    int preIn = configuration->inCount;
    layers = new vector<Layer*>();

    for (int neuronCount : *(neuronCounts)) 
    {
        Layer* layer = new Layer(preIn, neuronCount, configuration->initialWeightValue);
        layers->push_back(layer);
        preIn = neuronCount;
    }
}

MultiNetwork::~MultiNetwork()
{
    delete layers;
}

vector<double>* MultiNetwork::execute(vector<double>* inVector) 
{
    vector<double>* layerInVector = new vector<double>();

    for (double v : *inVector)
    {
        layerInVector->push_back(v);
    }

    for (Layer* layer : *layers) 
    {
        vector<Neuron*>* neurons = layer->getNeurons();

        for (Neuron* neuron : *neurons) 
        {
            neuron->inVector = layerInVector;
            neuron->generateOutput();
        }

        layerInVector->clear();

        for (Neuron* neuron : *neurons) 
        {
            layerInVector->push_back(neuron->output);
        }
    }

    return layerInVector;
}

void MultiNetwork::learn(bool showInfo) 
{
    int currentIteration = 1;
    double outputError = 1000000.0;

    while (outputError > 0 && currentIteration < configuration->maxLearningIterations) 
    {
        outputError = iteration();

        if (showInfo)
        {
            cout << currentIteration << ". error = " << outputError << endl;
        }

        currentIteration++;
    }

    cout << "MultiNetwork.learn finished with error = " << outputError << " iteration = " << currentIteration << endl;
}

double MultiNetwork::iteration() 
    {
        double totalErrorSum = 0.0;
        vector<TeachDataEntity>* learningData = configuration->teachData;

        double alphaValue = configuration->alpha;
        double speedValue = configuration->speed;
        int outputSize = configuration->neuronCounts->at(configuration->neuronCounts->size()-1);

        for (TeachDataEntity teachData : *(learningData)) 
        {
            // step 1 execute net with teach data
            vector<double>* actualOutput = execute(teachData.inp);
            vector<double>* expectedOutput = teachData.output;

            // step 2 generate sigma for last layer and update errorSum
            for (int j=0;j<outputSize;j++) 
            {
                layers->at(layers->size()-1)->getNeurons()->at(j)->sigma =
                    -1.0*actualOutput->at(j)*(1-actualOutput->at(j))*(expectedOutput->at(j)-actualOutput->at(j));
                totalErrorSum += (abs(expectedOutput->at(j)-actualOutput->at(j)) > 0.5)? 1.0 : 0.0;
            }

            // step 3 generate sigma for other layers
            for (int i=layers->size()-2;i>=0;i--) 
            {
                for (int j=0;j<layers->at(i)->getNeurons()->size();j++) 
                {
                    double currentSigma = 0.0;
                    double output = layers->at(i)->getNeurons()->at(j)->output;
                    double preSigma = output*(1-output);

                    for (Neuron* neuron : *(layers->at(i+1)->getNeurons())) 
                    {
                        currentSigma += neuron->weights->at(j) * neuron->sigma;
                    }

                    currentSigma = preSigma * currentSigma;

                    layers->at(i)->getNeurons()->at(j)->sigma = currentSigma;
                }
            }

            // step 4.1 generate delta
            for (int k=0;k<layers->size();k++) 
            {
                vector<double>* output = new vector<double>();
                if (k == 0) 
                {
                    for (double v : *(teachData.inp)) {
                        output->push_back(v);
                    }
                } 
                else 
                {
                    int outSize = layers->at(k-1)->getNeurons()->size();
                    output->clear();

                    for (int l=0;l<outSize;l++) 
                    {
                        output->push_back(layers->at(k-1)->getNeurons()->at(l)->output);
                    }
                }

                for (int j=0;j<layers->at(k)->getNeurons()->size();j++) 
                {
                    vector<double>* delta = layers->at(k)->getNeurons()->at(j)->delta;

                    for (int i=0;i<delta->size();i++) 
                    {
                        double currentSigma = layers->at(k)->getNeurons()->at(j)->sigma;
                        delta->at(i) = alphaValue*delta->at(i)+(1-alphaValue)*speedValue*currentSigma*output->at(i);
                    }
                }

                delete output;
            }

            // step 4.2 update weights
            for (Layer* layer : *(layers)) 
            {
                for (Neuron* neuron : *(layer->getNeurons())) 
                {
                    vector<double>* currentDelta = neuron->delta;

                    for (int i=0;i<neuron->inCount;i++) 
                    {
                        neuron->weights->at(i) = neuron->weights->at(i) - currentDelta->at(i);
                    }
                }
            }
        }

        return totalErrorSum;
    }
