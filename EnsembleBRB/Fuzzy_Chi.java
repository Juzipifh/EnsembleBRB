/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. Sánchez (luciano@uniovi.es)
    J. Alcalá-Fdez (jalcala@decsai.ugr.es)
    S. García (sglopez@ujaen.es)
    A. Fernández (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

package keel.Algorithms.Fuzzy_Rule_Learning.AdHoc.EnsembleBRB;

import java.io.IOException;
import org.core.*;

/**
 * <p>It contains the implementation of the Chi algorithm</p>
 *
 * @author Written by Alberto Fernández (University of Granada) 02/11/2007
 * @version 1.0
 * @since JDK1.5
 */
public class Fuzzy_Chi {

  myDataset train, oob;
  int nClasses, nLabels, combinationType, inferenceType, ruleWeight;
  int MaxCycle, NP;
  int[] Att;
  double F, CR;
  DataBase dataBase;
  RuleBase ruleBase;
  double classierWeight;

    /**
     * Configuration flags.
     */
    public static final int MINIMUM = 0;

    /**
     * Configuration flags.
     */
    public static final int PRODUCT = 1;

    /**
     * Configuration flags.
     */
    public static final int CF = 0;

    /**
     * Configuration flags.
     */
    public static final int PCF_IV = 1;

    /**
     * Configuration flags.
     */
    public static final int MCF = 2;

    /**
     * Configuration flags.
     */
    public static final int NO_RW = -1;

    public static final int Z_RW = 4;
    /**
     * Configuration flags.
     */
    public static final int PCF_II = 3;

    /**
     * Configuration flags.
     */
    public static final int WINNING_RULE = 0;

    /**
     * Configuration flags.
     */
    public static final int ADDITIVE_COMBINATION = 1;

  //We may declare here the algorithm's parameters

  private boolean somethingWrong = false; //to check if everything is correct.

  /**
   * Default constructor
   */
  public Fuzzy_Chi() {
  }

  /**
   * It reads the data from the input files (training, validation and test) and parse all the parameters
   * from the parameters array.
   * @param parameters parseParameters It contains the input files, output files and parameters
   */
  public Fuzzy_Chi(myDataset train, myDataset oob, int n_labels, int[] Att, int MaxCycle, int NP, double F, double CR) {

    //Now we parse the parameters
    this.nLabels = n_labels;
    this.train=train;
    this.Att=Att;
    this.MaxCycle=MaxCycle;
    this.NP=NP;
    this.F=F;
    this.CR=CR;
    this.oob=oob;
  }

  /**
   * It launches the algorithm
   */
  public void execute() {
    if (somethingWrong) { //We do not execute the program
      System.err.println("An error was found, the data-set have missing values");
      System.err.println("Please remove those values before the execution");
      System.err.println("Aborting the program");
      //We should not use the statement: System.exit(-1);
    }
    else {
      //We do here the algorithm's operations

      nClasses = train.getnClasses();
      int gen=0;
      Population pop=new Population(train, train, Att, nLabels,NP,F,CR);
      Individual indBest=new Individual(train,Att, nLabels);
      while(gen<=MaxCycle){
        pop.Mutation();
        pop.Crossover();
        pop.Selection();
        gen++;
        indBest=pop.saveBest();
      }
      dataBase=indBest.dataBase;
      ruleBase=indBest.ruleBase;
      System.out.println("Number of rules:"+ruleBase.getRuleNum()+"\n");
      System.out.println("Number of features:"+ruleBase.getAveFeatureNum()+"\n");
      classierWeight=1-indBest.caloobFitness(oob);
    }
  }
  /**
   * It returns the algorithm classification output given an input example
   * @param example double[] The input example
   * @return String the output generated by the algorithm
   */
  public double[] classificationOutput(double[] example) {
    double[] classOut = ruleBase.DisFRM(example);
    return classOut;
  }

  

}
