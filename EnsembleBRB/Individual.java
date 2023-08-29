package keel.Algorithms.Fuzzy_Rule_Learning.AdHoc.EnsembleBRB;

import java.util.Random;

public class Individual {
    double [][] RefValues;
    double [] attWeights;
    double [][] ruleConsequents;
    double [] ruleWeights;
    int[] Att;
    double fitness;
    double oobFit;

    DataBase dataBase;
    RuleBase ruleBase;

    myDataset train;

    public Individual(myDataset train, double [][] RefValues, int[] Att, double [] attWeights,double [][] ruleConsequents,double [] ruleWeights){
        this.RefValues=RefValues;
        this.attWeights=attWeights;
        this.ruleConsequents=ruleConsequents;
        this.ruleWeights=ruleWeights;
        this.Att=Att;
        this.dataBase=new DataBase(train, RefValues, Att, attWeights);
        this.ruleBase=new RuleBase(dataBase, Att, train.getNames(), train.getClasses());
        this.train=train;
        this.Att=Att;
        ruleBase.Generation();
        for(int i=0;i<ruleBase.getRuleNum();i++){
            Rule ri=ruleBase.ruleBase.get(i);
            ruleWeights[i]=ruleWeights[i]/max(ruleWeights);
            ri.setRweight(ruleWeights[i]);
            ri.setConsequent(ruleConsequents[i]);
        }
    }

    public Individual(myDataset train, int[] Att, int n_labels){
        Random r=new Random();
        this.train=train;
        this.Att=Att;
        int n_variables=train.getnInputs();
        this.RefValues=new double[n_variables][n_labels];
        this.attWeights=new double[n_variables];
        for (int i=0;i<Att.length;i++){
            double range=train.getRanges()[Att[i]][1]-train.getRanges()[Att[i]][0];
            RefValues[Att[i]][0]=train.getRanges()[Att[i]][0];
            RefValues[Att[i]][n_labels-1]=train.getRanges()[Att[i]][1];
            for(int j=1;j<n_labels-1;j++){
                RefValues[Att[i]][j]=train.getRanges()[Att[i]][0]+range*r.nextDouble();
            }
            attWeights[Att[i]]=r.nextDouble();
        }
        for (int i=0;i<n_variables;i++){
            attWeights[i]=attWeights[i]/max(attWeights);
        }
        dataBase=new DataBase( train, RefValues, Att, attWeights);
        ruleBase=new RuleBase(dataBase, Att, train.getNames(), train.getClasses());
        ruleBase.Generation();

        this.ruleWeights=new double [ruleBase.getRuleNum()];
        this.ruleConsequents=new double [ruleBase.getRuleNum()][train.getnClasses()];

        for(int i=0;i<ruleBase.getRuleNum();i++){
            ruleWeights[i]=r.nextDouble();
            double[] ruleCon=new double[train.getnClasses()];
            for(int j=0;j<ruleCon.length;j++){
                ruleCon[j]=r.nextDouble();
            }
            if(sum(ruleCon)>1){
                for(int j=0;j<ruleCon.length;j++){
                    ruleConsequents[i][j]=ruleCon[j]/sum(ruleCon);
                }
            }else{
                ruleConsequents[i]=ruleCon.clone();
            }
        }
        for(int i=0;i<ruleBase.getRuleNum();i++){
            Rule ri=ruleBase.ruleBase.get(i);
            ruleWeights[i]=ruleWeights[i]/max(ruleWeights);
            ri.setRweight(ruleWeights[i]);
            ri.setConsequent(ruleConsequents[i]);
        }
    }
    public double max(double[] X){
        double max=-1;
        for(int i=0;i<X.length;i++){
            if(X[i]>max){
                max=X[i];
            }
        }
        return max;
    }
    public double sum(double[] X){
        double sum=0;
        for(int i=0;i<X.length;i++){
            sum+=X[i];
        }
        return sum;
    }

    public double MSE(myDataset test){
        int hits=0;
        for( int i =0;i<test.getnData();i++){
            String output = new String("?");
            int classOut = ruleBase.FRM(test.getExample(i));
            if (classOut >= 0) {
                output = train.getOutputValue(classOut);
            }
            if (test.getOutputAsString(i).equalsIgnoreCase(output)){
                hits++;
            }
        }
        double mes=1-1.0*hits/test.size();
        return mes;
    }

    public Individual Mutation(Individual ind1,Individual ind2,double F){
        int[] newAtt=this.Att;
        Random r=new Random();
        double [][] newRefValues=new double[RefValues.length][RefValues[0].length];
        double [] newattWeights=new double[attWeights.length];
        for (int i=0;i<RefValues.length;i++){
            for(int j=0;j<RefValues[0].length;j++){
                double newx=this.RefValues[i][j]+F*(ind1.RefValues[i][j]-ind2.RefValues[i][j]);
                if(newx<this.RefValues[i][0]||newx>this.RefValues[i][this.RefValues[0].length-1]){
                    newx=this.RefValues[i][0]+r.nextDouble()*(this.RefValues[i][this.RefValues[0].length-1]-this.RefValues[i][0]);
                }
                newRefValues[i][j]=newx;
            }
            double newx=this.attWeights[i]+F*(ind1.attWeights[i]-ind2.attWeights[i]);
            if(newx<0||newx>1){
                newx=r.nextDouble();
            }
            newattWeights[i]=newx; 
        }

        double [][] newruleConsequents=new double[ruleBase.getRuleNum()][ruleConsequents[0].length];
        double [] newruleWeights=new double[ruleWeights.length];
        for(int i=0;i<ruleBase.getRuleNum();i++){
            double [] consequence=new double[ruleConsequents[0].length];
            for(int j=0;j<this.ruleConsequents[0].length;j++){
                double newx=this.ruleConsequents[i][j]+F*(ind1.ruleConsequents[i][j]-ind2.ruleConsequents[i][j]);
                if(newx<0||newx>1)
                    newx=r.nextDouble();
                consequence[j]=newx;
            }
            if(sum(consequence)>1){
                for(int j=0;j<this.ruleConsequents[0].length;j++){
                    newruleConsequents[i][j]=consequence[j]/sum(consequence);
                }
            }else{
                newruleConsequents[i]=consequence.clone();
            }
            double newx=this.ruleWeights[i]+F*(ind1.ruleWeights[i]-ind2.ruleWeights[i]);
            if(newx<0||newx>1){
                newx=r.nextDouble();
            }
            newruleWeights[i]=newx;   
        }
        return new Individual(this.train, newRefValues, newAtt, newattWeights, newruleConsequents, newruleWeights);
    }

    public Individual Crossover(Individual ind1, double CR){
        int[] newAtt=this.Att;
        Random r=new Random();
        double [][] newRefValues=new double[RefValues.length][RefValues[0].length];
        double [] newattWeights=new double[attWeights.length];
        for (int i=0;i<RefValues.length;i++){
            for(int j=0;j<RefValues[0].length;j++){
                if(r.nextDouble()<CR)
                newRefValues[i][j]=ind1.RefValues[i][j];
                else
                newRefValues[i][j]=this.RefValues[i][j];
            }
            if(r.nextDouble()<CR)
            newattWeights[i]=ind1.attWeights[i];
            else
            newattWeights[i]=this.attWeights[i];
        }

        double [][] newruleConsequents=new double[ruleBase.getRuleNum()][ruleConsequents[0].length];
        double [] newruleWeights=new double[ruleWeights.length];
        for(int i=0;i<ruleBase.getRuleNum();i++){
            double [] consequence=new double[ruleConsequents[0].length];
            for(int j=0;j<this.ruleConsequents[0].length;j++){
                if(r.nextDouble()<CR)
                consequence[j]=ind1.ruleConsequents[i][j];
                else
                consequence[j]=this.ruleConsequents[i][j];
            }
            if(sum(consequence)>1){
                for(int j=0;j<this.ruleConsequents[0].length;j++){
                    newruleConsequents[i][j]=consequence[j]/sum(consequence);
                }
            }else{
                newruleConsequents[i]=consequence.clone();
            }
            if(r.nextDouble()<CR)
            newruleWeights[i]=ind1.ruleWeights[i];
            else
            newruleWeights[i]=this.ruleWeights[i];
        }
        return new Individual(this.train,newRefValues,newAtt, newattWeights, newruleConsequents, newruleWeights);
    }

    public void setFitness(double fit){
        this.fitness=fit;
    }

    public double calFitness(myDataset train){
        this.fitness=this.MSE(train);
        return this.fitness;
    }

    public double getFitness(){
        return this.fitness;
    }

    public void setoobFitness(double fit){
        this.oobFit=fit;
    }

    public double caloobFitness(myDataset test){
        this.oobFit=this.MSE(test);
        return this.oobFit;
    }

    public double getoobFitness(){
        return this.oobFit;
    }
    
}
