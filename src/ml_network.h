#include <vector>

using namespace std;

struct TeachDataEntity 
{
	vector<double>* inp;
	vector<double>* output;
};

struct NetConfiguration 
{
    int inCount;
    vector<int>* neuronCounts;
    int maxLearningIterations;
    double initialWeightValue;
    double alpha, speed;
    vector<TeachDataEntity>* teachData;
};

class Neuron 
{
	public:
	int inCount;
	vector<double>* inVector;
	vector<double>* weights;
	double net;
	double output;
	double sigma;
	vector<double>* delta;
	double weightOffset;

	Neuron(int inCount, double initialWeightValue);
    ~Neuron();
	double generateOutput();
};

class Layer 
{
    private: 
    vector<Neuron*>* neurons;

    public: 
    Layer(int inCount, int neuronLayerCount, double initialWeightValue);
    ~Layer();
    vector<Neuron*>* getNeurons();
};

class MultiNetwork 
{
    private:
    NetConfiguration* configuration;
    vector<Layer*>* layers;

    public: 
    MultiNetwork(NetConfiguration* configuration);
    ~MultiNetwork();
    vector<double>* execute(vector<double>* inVector);
    void learn(bool showInfo);
    double iteration();
};
