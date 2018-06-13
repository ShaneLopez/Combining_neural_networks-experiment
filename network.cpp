#include "network.h"

void network::init(double ulr, int inputnum, int hiddennum, int outputnum)
{
    srand((unsigned)time(NULL));
    vector<double> temp;
    ihweights.clear();
    howeights.clear();
    inputs.clear();
    hidden.clear();
    outputs.clear();

    lr = ulr;

    for(int i = 0; i < inputnum; i++)
    {
        temp.clear();
        for(int j = 0; j < hiddennum; j++)
        {
            temp.push_back(-5 + ((double)rand() / RAND_MAX) * (5 + 5));
        }
        ihweights.push_back(temp);
        inputs.push_back(0);
    }

    for(int i = 0; i < hiddennum; i++)
    {
        temp.clear();
        for(int j = 0; j < outputnum; j++)
        {
            temp.push_back(-5 + ((double)rand() / RAND_MAX) * (5 + 5));
        }
        howeights.push_back(temp);
        hidden.push_back(0);
    }

    for(int i = 0; i < outputnum; i++)
    {
        outputs.push_back(0);
    }
}

void network::forwardpass(vector<double> uinput)
{
    if(uinput.size() > inputs.size())
    {
        cout << "too many inputs " << uinput.size() << " " << inputs.size() << endl;
        return;
    }

    double total;

    for(int i = 0; i < uinput.size(); i++)
    {
        inputs[i] = uinput[i];
    }

    for(int i = 0; i < hidden.size(); i++)
    {
        total = 0;
        for(int j = 0; j < inputs.size(); j++)
        {
            total += ihweights[j][i] * inputs[j];
        }
        hidden[i] = tanh(total);
    }

    for(int i = 0; i < outputs.size(); i++)
    {
        total = 0;
        for(int j = 0; j < hidden.size(); j++)
        {
            total += howeights[j][i] * hidden[j];
        }
        outputs[i] = tanh(total);
    }
}

void network::backprop(vector<double> error)
{
    double totalerror = 0;
    double sumweights = 0;
    vector<double> reterror;

    if(error.size() > 1)
    {
        for(int i = 0; i < error.size(); i++)
        {
            totalerror += error[i];
        }

        totalerror = pow(totalerror, 2) / error.size();
    }
    else
    {
        totalerror = error[0];
    }

    for(int i = 0; i < outputs.size(); i++)
    {
        for(int j = 0; j < hidden.size(); j++)
        {
            howeights[j][i] -= lr * error[i] * hidden[j];

            if(howeights[j][i] > 5)
                howeights[j][i] = 5;
            else if(howeights[j][i] < -5)
                howeights[j][i] = -5;
        }
    }

    for(int i = 0; i < hidden.size(); i++)
    {
        sumweights = 0;
        for(int k = 0; k < outputs.size(); k++)
        {
            sumweights += howeights[i][k];
        }

        for(int j = 0; j < inputs.size(); j++)
        {
            ihweights[j][i] -= lr * sumweights * totalerror * (1 - hidden[i] * hidden[i]) * inputs[j];

            if(ihweights[j][i] > 5)
                ihweights[j][i] = 5;
            else if(ihweights[j][i] < -5)
                ihweights[j][i] = -5;
        }
    }
}

void network::getoutputs(vector<double>& uoutput)
{
    uoutput.clear();
    for(int i = 0; i < outputs.size(); i++)
    {
        uoutput.push_back(outputs[i]);
    }
}

void network::setweights(vector< vector<double> > uihweights, vector< vector<double> > uhoweights)
{
    for(int i = 0; i < ihweights.size(); i++)
    {
        for(int j = 0; j < ihweights[i].size(); j++)
        {
            ihweights[i][j] = uihweights[i][j];
        }
    }

    for(int i = 0; i < howeights.size(); i++)
    {
        for(int j = 0; j < howeights[i].size(); j++)
        {
            howeights[i][j] = uhoweights[i][j];
        }
    }
}

void network::getweights(vector< vector<double> >& uihweights, vector< vector<double> >& uhoweights)
{
    uihweights.clear();
    uhoweights.clear();
    vector<double> temp;
    for(int i = 0; i < ihweights.size(); i++)
    {
        temp.clear();
        for(int j = 0; j < ihweights[i].size(); j++)
        {
            temp.push_back(ihweights[i][j]);
        }
        uihweights.push_back(temp);
    }

    for(int i = 0; i < howeights.size(); i++)
    {
        temp.clear();
        for(int j = 0; j < howeights[i].size(); j++)
        {
            temp.push_back(howeights[i][j]);
        }
        uhoweights.push_back(temp);
    }
}
