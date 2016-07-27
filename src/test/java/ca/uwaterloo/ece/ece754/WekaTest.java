package ca.uwaterloo.ece.ece754;

import static org.junit.Assert.assertTrue;

import java.io.*;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.junit.Test;

import ca.uwaterloo.ece.ece754.utils.Util;
import cs.uwaterloo.ece.ece754.TrainTest;
import cs.uwaterloo.ece.ece754.computePvalue;
import cs.uwaterloo.ece.ece754.evalRes;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.converters.AbstractFileSaver;
import weka.core.converters.CSVSaver;

public class WekaTest {
    @Test public void testWeka() {
    	TrainTest ts=new TrainTest();
    	//evalRes res=ts.getTestRes("jackrabbit", 10, 2,new String[]{"-R","12-13"});
    	String[] name= new String[]{"jackrabbit","linux","postgresql","jdt","lucene","xorg"};
    	int[] num=new int[]{10,4,7,6,8,6};
    	computePvalue cp=new computePvalue();
    	File tempRes=new File("data/res");
    	File fname=new File("data/10new_prf1.csv");
    	for(int i=0;i<6;i++){
    		evalRes res=ts.getTestRes(name[i], num[i], 10, new String[]{"-R","12-13"},fname);	 
    	}
//    	MannWhitneyUTest mtest=new MannWhitneyUTest();
//    	double[] x1=new double[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
//    	double[] x2=new double[]{7,6,5,4,3,2,1};
//    	double[] x3=new double[]{7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};;
//    	System.out.println(mtest.mannWhitneyUTest(x1, x2));
//    	System.out.println(mtest.mannWhitneyUTest(x1, x3));
//    	for(int j=16;j<=16;j++){
//        	File fname=new File("data/new_"+j+".csv");
//        	for(int i=0;i<6;i++){
//        		evalRes res=ts.getTestRes(name[i], num[i], j, new String[]{"-R","12-13"},fname);	 
//            }
//    	}

//    	for(int j=10;j<=15;j++){
//        	File fname=new File("data/PRF_"+j+"cleanedData.csv");
//        	int i=0;
//	    System.out.println(name[i]);
//		   evalRes res=ts.getTestRes(name[i], num[i], j, new String[]{"-R","12-13"},fname);	   	
//		   i=4;
//		   System.out.println(name[i]);
//   		   res=ts.getTestRes(name[i], num[i], j, new String[]{"-R","12-13"},fname);	
//    	}


    }
}
