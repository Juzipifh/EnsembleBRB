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

import java.util.Arrays;

import org.core.Files;

/**
 * <p>This class contains the representation of a Fuzzy Data Base</p>
 *
 * @author Written by Alberto Fernández (University of Granada) 28/10/2007
 * @author Modified by Alberto Fernández (University of Granada) 12/11/2008
 * @version 1.1
 * @since JDK1.5
 */
public class DataBase {
    int n_variables;
    int n_labels;
    double[][] RefValues;
    double [] attWeights;
    Fuzzy[][] dataBase;
    String[] names;
    int[] Att;

    /**
     * Default constructor
     */
    public DataBase() {
    }

    /**
     * Constructor with parameters. It performs a homegeneous partition of the input space for
     * a given number of fuzzy labels.
     * @param n_variables int Number of input variables of the problem
     * @param n_labels int Number of fuzzy labels
     * @param rangos double[][] Range of each variable (minimum and maximum values)
     * @param names String[] Labels for the input attributes
     */
    public DataBase(myDataset train, double[][] RefValues, int[] Att, double [] attWeights) {
        this.n_variables = attWeights.length;
        this.n_labels = RefValues[0].length;
        this.RefValues=RefValues;
        this.attWeights=attWeights;
        this.Att=Att;
        dataBase = new Fuzzy[n_variables][n_labels];
        this.names = train.getNames().clone();
        for (int i = 0; i < Att.length; i++) {
            Arrays.sort(RefValues[Att[i]]);
            for (int j=0;j<n_labels;j++){
                dataBase[Att[i]][j]=new Fuzzy();
                dataBase[Att[i]][j].label=j;
                dataBase[Att[i]][j].name=new String("L_" + Att[i]);
                dataBase[Att[i]][j].attWeight=attWeights[Att[i]];
                dataBase[Att[i]][j].y=1.0;
                if(j==0){
                    dataBase[Att[i]][j].x0=RefValues[Att[i]][j]-(RefValues[Att[i]][j+1]-RefValues[Att[i]][j]);
                }else{
                    dataBase[Att[i]][j].x0=RefValues[Att[i]][j-1];
                }
                dataBase[Att[i]][j].x1=RefValues[Att[i]][j];
                if(j==n_labels-1){
                    dataBase[Att[i]][j].x3=RefValues[Att[i]][j]+(RefValues[Att[i]][j]-RefValues[Att[i]][j-1]);
                }else{
                    dataBase[Att[i]][j].x3=RefValues[Att[i]][j+1];
                }
            }
        }
    }

    /**
     * It returns the number of input variables
     * @return int the number of input variables
     */
    public int numVariables() {
        return n_variables;
    }

    /**
     * It returns the number of fuzzy labels
     * @return int the number of fuzzy labels
     */
    public int numLabels() {
        return n_labels;
    }

    /**
     * It computes the membership degree for a input value
     * @param i int the input variable id
     * @param j int the fuzzy label id
     * @param X double the input value
     * @return double the membership degree
     */
    public double membershipFunction(int i, int j, double X) {
        return dataBase[i][j].Fuzzify(X);
    }

    /**
     * It makes a copy of a fuzzy label
     * @param i int the input variable id
     * @param j int the fuzzy label id
     * @return Fuzzy a copy of a fuzzy label
     */
    public Fuzzy clone(int i, int j) {
        return dataBase[i][j].clone();
    }

    /**
     * It prints the Data Base into an string
     * @return String the data base
     */
    public String printString() {
        String cadena = new String(
                "@Using Triangular Membership Functions as antecedent fuzzy sets\n");
        cadena += "@Number of Labels per variable: " + n_labels + "\n";
        for (int i = 0; i < n_variables; i++) {
            //cadena += "\nVariable " + (i + 1) + ":\n";
            cadena += "\n" + names[i] + ":\n";
            for (int j = 0; j < n_labels; j++) {
                cadena += " L_" + (j + 1) + ": (" + dataBase[i][j].x0 +
                        "," + dataBase[i][j].x1 + "," + dataBase[i][j].x3 +
                        ")\n";
            }
        }
        return cadena;
    }

    /**
     * It writes the Data Base into an output file
     * @param filename String the name of the output file
     */
    public void writeFile(String filename) {
        String outputString = new String("");
        outputString = printString();
        Files.writeFile(filename, outputString);
    }

}

