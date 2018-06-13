#include "network.h"

struct smartnet
{
    network net;
    bool learned[2] = {false};
    bool empty = true;
};

int main()
{
    network shorttermmem;
    smartnet longtermmem[3];
    int inputsize = 3;
    int hiddensize = 7;
    int outputsize = 1;
    vector< vector<double> > ihweights;
    vector< vector<double> > howeights;
    vector< vector<double> > tempweights;
    vector<double>temp;
    vector< vector<double> > inputs{{-1,-1,1},{-1,1,1},{1,-1,1},{1,1,1}};
    vector<double> outputs;
    vector< vector<double> > expoutputs{{-1,-1,-1,1,1},{1,-1,-1,1,1},{-1,1,1,1,1},{1,-1,-1,-1,1},{1,1,1,-1,1},{-1,1,1,-1,1}};
    vector<double> error;
    vector<int> learnedindex;
    vector<double> learnedtemp;
    vector< vector<double> > learnedexpoutput;
    double stoperror = 1;
    double sumerror;
    double localsum;
    double localsquare;
    int check;
    int cur;
    int counter;
    int choice;
    int shake = 0;
    double errorcheck = 1;
    int itcount = 0;
    double range = 5;

    shorttermmem.init(0.1, inputsize, hiddensize, outputsize);
    for(int i = 0; i < 3; i++)
    {
        longtermmem[i].net.init(0.1, 5, 8, (inputsize * hiddensize) + (hiddensize * outputsize));
    }

    for(int i = 0; i < expoutputs.size(); i++)
    {
        cur = ceil((double)(i + 1) / 2.0) - 1;
        longtermmem[cur].net.forwardpass(expoutputs[i]);
        longtermmem[cur].net.getoutputs(outputs);

        counter = 0;
        for(int j = 0; j < inputsize; j++)
        {
            for(int k = 0; k < hiddensize; k++)
            {
                temp.push_back(outputs[counter] * range);
                counter++;
            }
            ihweights.push_back(temp);
            temp.clear();
        }
        for(int j = 0; j < hiddensize; j++)
        {
            for(int k = 0; k < outputsize; k++)
            {
                temp.push_back(outputs[counter] * range);
                counter++;
            }
            howeights.push_back(temp);
            temp.clear();
        }
        shorttermmem.setweights(ihweights, howeights);

        stoperror = 1;
        sumerror = 0;
        check = 0;

        //clock_t begin = clock();

        itcount = 0;
        while(stoperror > 0.05)
        {
            choice = rand() % inputs.size();
            shorttermmem.forwardpass(inputs[choice]);
            shorttermmem.getoutputs(outputs);
            for(int j = 0; j < outputs.size(); j++)
            {
                error.push_back(outputs[j] - expoutputs[i][choice]);
                sumerror += fabs(error[j]);
            }
            check++;
            shorttermmem.backprop(error);
            error.clear();
            itcount++;

            if(check == 50)
            {
                //shake++;
                stoperror = sumerror / check;
                sumerror = 0;
                check = 0;
                /*if(shake == 20)
                {
                    shake = 0;
                    shorttermmem.init(0.1,inputsize, hiddensize, outputsize);
                }*/
            }
        }
        cout << itcount << endl;
        /*clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        printf("%f\n", elapsed_secs);*/

        //longtermmem[cur].net.getoutputs(outputs);
        shorttermmem.getweights(ihweights, howeights);
        for(int j = 0; j < ihweights.size(); j++)
        {
            for(int k = 0; k < ihweights[j].size(); k++)
            {
                temp.push_back(ihweights[j][k] / range);
            }
        }
        for(int j = 0; j < howeights.size(); j++)
        {
            for(int k = 0; k < howeights[j].size(); k++)
            {
                temp.push_back(howeights[j][k] / range);
            }
        }
        longtermmem[cur].learned[i - (cur * 2)] = true;
        longtermmem[cur].empty = false;

        for(int j = 0; j < 2; j++)
        {
            learnedtemp.clear();
            if(longtermmem[cur].learned[j])
            {
                learnedindex.push_back((cur * 2) + j);
                longtermmem[cur].net.forwardpass(expoutputs[(cur * 2) + j]);
                longtermmem[cur].net.getoutputs(learnedtemp);
            }
            if(!learnedtemp.empty() && j != i - (cur * 2))
            {
                learnedexpoutput.push_back(learnedtemp);
            }
            else if(j == i - (cur * 2))
            {
                learnedexpoutput.push_back(temp);
            }
        }

        stoperror = 1;
        sumerror = 0;
        check = 0;
        shake = 0;
        localsum = 0;

        while(stoperror > 1 / (10 * errorcheck))
        {
            choice = rand() % learnedindex.size();
            longtermmem[cur].net.forwardpass(expoutputs[learnedindex[choice]]);
            longtermmem[cur].net.getoutputs(outputs);
            for(int j = 0; j < outputs.size(); j++)
            {
                error.push_back(outputs[j] - learnedexpoutput[choice][j]);
                localsum += fabs(error[j]);
            }
            localsquare = localsum / outputs.size();
            sumerror += localsquare;
            localsum = 0;
            check++;
            longtermmem[cur].net.backprop(error);
            error.clear();

            if(check == 50)
            {
                //shake++;
                stoperror = sumerror / check;
                sumerror = 0;
                check = 0;
                /*if(shake == 20)
                {
                    longtermmem[cur].net.init(0.1, 5, 10, (inputsize * hiddensize) + (hiddensize * outputsize));
                    shake = 0;
                }*/
            }
        }
        errorcheck *= 10;
        /*if(i == 5)
        {
            longtermmem[cur].net.forwardpass(expoutputs[5]);
            longtermmem[cur].net.getoutputs(outputs);
            for(int j = 0; j < outputs.size(); j++)
            {
                cout << outputs[j] << " " << temp[j] << endl;
            }
        }*/
        if(longtermmem[cur + 1].empty && cur < 2)
        {
            longtermmem[cur].net.getweights(ihweights,howeights);
            longtermmem[cur + 1].net.setweights(ihweights,howeights);
        }
        if(learnedexpoutput.size() == 2)
            errorcheck = 1;
        learnedexpoutput.clear();
        learnedindex.clear();
        ihweights.clear();
        howeights.clear();
        temp.clear();
        cout << "-" << endl;
    }
    cout << endl;
    cout << endl;

    longtermmem[0].net.forwardpass(expoutputs[1]);
    longtermmem[0].net.getoutputs(outputs);

    counter = 0;
    for(int j = 0; j < inputsize; j++)
    {
        for(int k = 0; k < hiddensize; k++)
        {
            temp.push_back(outputs[counter] * range);
            counter++;
        }
        ihweights.push_back(temp);
        temp.clear();
    }
    for(int j = 0; j < hiddensize; j++)
    {
        for(int k = 0; k < outputsize; k++)
        {
            temp.push_back(outputs[counter] * range);
            counter++;
        }
        howeights.push_back(temp);
        temp.clear();
    }
    shorttermmem.setweights(ihweights, howeights);

    for(int i = 0; i < inputs.size(); i++)
    {
        shorttermmem.forwardpass(inputs[i]);
        shorttermmem.getoutputs(outputs);
        /*if(outputs[0] > 0)
            cout << 1 << " ";
        else
            cout << 0 << " ";*/
            cout << outputs[0] << " ";
    }

    cout << endl;
    return 0;
}
