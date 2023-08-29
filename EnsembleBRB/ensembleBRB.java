package keel.Algorithms.Fuzzy_Rule_Learning.AdHoc.EnsembleBRB;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import org.core.*;

//paper: Ensemble Belief Rule-Based Model for complex system classification and prediction

public class ensembleBRB {
    myDataset train, val, test;
    String outputTr, outputTst, fileDB, fileRB;

    int nClasses, nLabels;
    int MaxCycle, numAtt, NP, numWeak;
    double F, CR;

    ArrayList<Fuzzy_Chi> fuzzyChis;

    private boolean somethingWrong = false; //to check if everything is correct.

    public ensembleBRB(){
    }

    public ensembleBRB(parseParameters parameters) {

        train = new myDataset();
        val = new myDataset();
        test = new myDataset();
        try {
          System.out.println("\nReading the training set: " +
                             parameters.getTrainingInputFile());
          train.readClassificationSet(parameters.getTrainingInputFile(), true);
          System.out.println("\nReading the validation set: " +
                             parameters.getValidationInputFile());
          val.readClassificationSet(parameters.getValidationInputFile(), false);
          System.out.println("\nReading the test set: " +
                             parameters.getTestInputFile());
          test.readClassificationSet(parameters.getTestInputFile(), false);
        } catch (IOException e) {
          System.err.println(
              "There was a problem while reading the input data-sets: " +
              e);
          somethingWrong = true;
        }
    
        //We may check if there are some numerical attributes, because our algorithm may not handle them:
        //somethingWrong = somethingWrong || train.hasNumericalAttributes();
        somethingWrong = somethingWrong || train.hasMissingAttributes();
    
        outputTr = parameters.getTrainingOutputFile();
        outputTst = parameters.getTestOutputFile();
    
        fileDB = parameters.getOutputFile(0);
        fileRB = parameters.getOutputFile(1);
    
        //Now we parse the parameters
        nLabels = Integer.parseInt(parameters.getParameter(0));
        numAtt=Integer.parseInt(parameters.getParameter(1));
        numWeak=Integer.parseInt(parameters.getParameter(2));
        MaxCycle = Integer.parseInt(parameters.getParameter(3));
        NP=Integer.parseInt(parameters.getParameter(4));
        F=Double.parseDouble(parameters.getParameter(5));
        CR=Double.parseDouble(parameters.getParameter(6));
      }

      public void execute() {
        if (somethingWrong) { //We do not execute the program
          System.err.println("An error was found, the data-set have missing values");
          System.err.println("Please remove those values before the execution");
          System.err.println("Aborting the program");
          //We should not use the statement: System.exit(-1);
        }
        else {
            nClasses=train.getnClasses();
            int weakDataNum=train.getnData();
            // weakDataNum=train.getnData()/10;
            if(train.getnData()<=200){
                weakDataNum=train.getnData();
            }else if(train.getnData()>=200&&train.getnData()<=500){
                weakDataNum=train.getnData()/2;
            }else if(train.getnData()>=500&&train.getnData()<=1000){
                weakDataNum=train.getnData()/5;
            }else if(train.getnData()>=1000&&train.getnData()<=2000){
                weakDataNum=train.getnData()/10;
            }else if(train.getnData()>=2000&&train.getnData()<=10000){
                weakDataNum=train.getnData()/50;
            }else{
                weakDataNum=train.getnData()/100;
            }
            fuzzyChis=new ArrayList<Fuzzy_Chi>();
            for(int i=0;i<numWeak;i++){
                Random r=new Random();
                ArrayList<Integer> att=new ArrayList<>();
                int a=0;
                while(a<numAtt){
                    int index=r.nextInt(train.getnInputs());
                    if(!att.contains(index)){
                        att.add(index);
                        a++;
                    }
                }
                int[] Att=new int[numAtt];
                for(a=0;a<numAtt;a++){
                    Att[a]=att.get(a);
                }
                myDataset weakTrain, weakOOB;
                if(weakDataNum<train.getnData()){
                    myDataset[] datasplit=train.getRandomNSamples(weakDataNum);
                    weakTrain=datasplit[0];
                    weakOOB=datasplit[1];

                }else{
                    weakTrain=train;
                    weakOOB=train;
                }
                Fuzzy_Chi weakChiI=new Fuzzy_Chi(weakTrain, weakOOB, nLabels, Att, MaxCycle, NP, F, CR);
                weakChiI.execute();
                fuzzyChis.add(weakChiI);
            }
            writeFile(this.fileRB);
            double accTra = doOutput(this.val, this.outputTr);
            double accTst = doOutput(this.test, this.outputTst);
      
            System.out.println("Accuracy obtained in training: "+accTra);
            System.out.println("Accuracy obtained in test: "+accTst);
            System.out.println("Algorithm Finished");
        }
      }
    
    public String classificationOutput(double[] example){
        String output = new String("?");
        double[] outdistribution=new double[nClasses];
        for(int i=0;i<numWeak;i++){
            Fuzzy_Chi weaki=fuzzyChis.get(i);
            double[] clasi=weaki.classificationOutput(example);
            for(int j=0;j<nClasses;j++){
                outdistribution[j]+=clasi[j]*weaki.classierWeight;
            }
        }
        double max=0;
        int clas=-1;
        for(int i=0;i<nClasses;i++){
            if(outdistribution[i]>max){
                max=outdistribution[i];
                clas=i;
            }
        }
        if(clas>=0){
            output = train.getOutputValue(clas);
        }
        return output;
    }

      
  /**
   * It generates the output file from a given dataset and stores it in a file
   * @param dataset myDataset input dataset
   * @param filename String the name of the file
   *
   * @return The classification accuracy
   */
  private double doOutput(myDataset dataset, String filename) {
    String output = new String("");
    int hits = 0;
    output = dataset.copyHeader(); //we insert the header in the output file
    //We write the output for each example
    for (int i = 0; i < dataset.getnData(); i++) {
      //for classification:
      String classOut = this.classificationOutput(dataset.getExample(i));
      output += dataset.getOutputAsString(i) + " " + classOut + "\n";
      if (dataset.getOutputAsString(i).equalsIgnoreCase(classOut)){
        hits++;
      }
    }
    Files.writeFile(filename, output);
    return (1.0*hits/dataset.size());
  }

  public void writeFile(String filename) {
    String outputString = new String("");
    outputString = printString();
    Files.writeFile(filename, outputString);
}

private String printString() {
    String str="";
    for(int i=0;i<fuzzyChis.size();i++){
        str+=i+"th weak BRB:\n";
        str+=fuzzyChis.get(i).ruleBase.printString();
        str+="\n";
    }
    return str;
}

}
