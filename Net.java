import java.io.*;
import java.util.Random;
import java.util.Scanner;

public class Net {

    private int layers[];

    private boolean isBias;

    private String file;

    private Neuron Neurons[];

    private Synapse synapse[];

    public void trainNet(String fileName, int numLoop, float learningRate) throws IOException{
        Scanner scanner = new Scanner(new FileReader(fileName));
        float expected[] = new float[layers[layers.length - 1]];
        float in_arr[] = new float[layers[0] - ((isBias) ? 1 : 0)];
        for(int i = 0, n = 0; i < numLoop; i++, n++) {
            if(!scanner.hasNext()) scanner = new Scanner(new FileReader(fileName));

            if (scanner.next().equals("input:"))
                for (int j = 0; j < in_arr.length; j++)
                    in_arr[j] = Float.parseFloat(scanner.next());

            if (scanner.next().equals("output:"))
                for (int j = 0; j < expected.length; j++)
                    expected[j] = Float.parseFloat(scanner.next());

            train(in_arr, expected, learningRate);
        }
        saveWeights();
        scanner.close();
    }

    public void createNet(int layers[], boolean isBias, String file) throws IOException{
        makeNet(layers, isBias, file);
        setRandWeights();
        saveWeights();
    }

    public void initNet(int layers[], boolean isBias, String file) throws IOException{
        makeNet(layers, isBias, file);
        initWeights();
    }

    public void train(float inputArray[], float expected[], float learningRate){
        float errArray[] = new float[Neurons.length - layers[0]];
        float actual[];
        float weightDelta;
        actual = getResult(inputArray);
        for(int j = 0, m = getPosNeuron(layers.length - 1); j < layers[layers.length - 1]; j++, m++) errArray[m - layers[0]] = actual[j] - expected[j];
        for(int j = Neurons.length - 1; j >= getPosNeuron(1); j--){
            if(Neurons[j].outS.length > 0){
                float sum_err = 0;
                for(int m = 0; m < Neurons[j].outS.length; m++)
                    sum_err += errArray[synapse[Neurons[j].outS[m]].out - layers[0]] * synapse[Neurons[j].outS[m]].weight;
                errArray[j - layers[0]] = sum_err;
            }
            weightDelta = errArray[j - layers[0]] * getDerivative(Neurons[j].value);
            for(int m = 0; m < Neurons[j].inS.length; m++){
                synapse[Neurons[j].inS[m]].weight -= weightDelta * Neurons[synapse[Neurons[j].inS[m]].in].value * learningRate;
            }
        }
    }

    public void saveWeights() throws IOException {
        FileWriter fileWriter = new FileWriter(file);
        for(int i = 0; i < synapse.length; i++)
            fileWriter.write(Float.toString(synapse[i].weight) + "\n");
        fileWriter.close();
    }

    public float[] getResult(float in[]){
        float result[] = new float[layers[layers.length - 1]];
        for(int i = 0, iL = 0; i < Neurons.length; i++){
            if( !isBias && iL == 0 || isBias && iL == 0 && i < layers[0] - 1) Neurons[i].value = in[i];
            else{
                float sum = 0;
                for(int j = 0; j < Neurons[i].inS.length; j++)
                    sum += synapse[Neurons[i].inS[j]].weight * Neurons[synapse[Neurons[i].inS[j]].in].value;
                Neurons[i].value = activationFoo(sum);
            }
            if(i == getPosNeuron(iL) + layers[iL] - 1) iL++;
        }
        for(int i = 0, j = getPosNeuron(layers.length - 1); i < layers[layers.length - 1]; i++, j++) result[i] = Neurons[j].value;
        return result;
    }

    private int getNumSynapse(){
        int sum = 0;
        if(!isBias) {
            for (int i = 0; i < layers.length - 1; i++)
                sum += layers[i] * layers[i + 1];
        }
        else{
            for(int i = 0; i < layers.length - 1; i++)
                sum += layers[i] * ((i != layers.length - 2) ? (layers[i + 1] - 1) : layers[i + 1]);
        }
        return sum;
    }

    private int getNumNeurons(){
        int sum = 0;
        for(int i = 0; i < layers.length; i++) sum += layers[i];
        return sum;
    }

    private int getPosNeuron(int layer){
        int sum = 0;
        for(int i = 0; i < layer; i++) sum += layers[i];
        return sum;
    }

    private void makeNet(int layers[], boolean isBias, String file){
        this.layers = layers;
        this.isBias = isBias;
        this.file = file;

        if(isBias) for(int i = 0; i < layers.length - 1; i++) layers[i]++;

        Neurons = new Neuron[getNumNeurons()];
        synapse = new Synapse[getNumSynapse()];

        for(int i = 0, iS = 0, oS = 0, iL = 0; i < Neurons.length; i++){
            Neurons[i] = new Neuron();
            Neurons[i].id = i;

            if(isBias) Neurons[i].outS = new int[(iL < layers.length - 1) ? ((iL == layers.length - 2) ? layers[iL + 1] : layers[iL + 1] - 1) : 0];
            else Neurons[i].outS = new int[(iL < layers.length - 1) ? layers[iL + 1] : 0];

            if(isBias && iL != layers.length - 1 && i == getPosNeuron(iL) + layers[iL] - 1){
                Neurons[i].inS = new int[0];
                Neurons[i].value = 1;
            }
            else {
                Neurons[i].inS = new int[(iL > 0) ? layers[iL - 1] : 0];
            }
            for(int j = 0; j < Neurons[i].inS.length; j++){
                synapse[iS] = new Synapse();
                synapse[iS].in = getPosNeuron(iL - 1) + j;
                synapse[iS].out = i;
                Neurons[i].inS[j] = iS;
                Neurons[synapse[iS].in].outS[oS] = iS;
                iS++;
            }
            oS++;
            if(i == getPosNeuron(iL) + layers[iL] - 1) {
                iL++;
                oS = 0;
            }
        }
    }

    private void initWeights() throws IOException{
        FileReader fileReader = new FileReader(file);
        Scanner scanner = new Scanner(fileReader);
        for(int i = 0; i < synapse.length && scanner.hasNext(); i++)
            synapse[i].weight = Float.parseFloat(scanner.nextLine());
        fileReader.close();
    }

    private void setRandWeights(){
        Random random = new Random();
        for(int i = 0; i < synapse.length; i++)
            synapse[i].weight = random.nextFloat();
    }

    private float activationFoo(float x){
        return (float) (1 / (1 + Math.pow(2.71828, -x)));
    }

    private float getDerivative(float x){
        return (x * (1 - x));
    }
    
 }
 
 public class Neuron {
    int id;
    int inS[];
    int outS[];
    float value;
 }
 
 public class Synapse {
    int in;
    int out;
    float weight;
}


