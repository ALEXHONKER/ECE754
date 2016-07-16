package ca.uwaterloo.ece.ece754;

import static org.junit.Assert.assertTrue;

import java.io.*;

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
    	File fname=new File("data/usefirst16fetures.csv");
//    	try {
			//BufferedWriter bw=new BufferedWriter(new FileWriter(tempRes));
	    	for(int i=0;i<6;i++){
	    		System.out.println(name[i]);
//	    		System.out.println("index:"+i);
	        	//evalRes res=ts.getTestRes(name[i], num[i], 4,new String[]{"-R","12-13"});
//	    		for(int j=0;j<num[i]-1;j++){
		    		//cp.compute(name[i], num[i], 7, fname);
		    		evalRes res=ts.getTestRes(name[i], num[i], 10, null,fname);	 
		    		//evalRes res=ts.getTestRes(name[i], num[i], 7,null,fname);	
		    		//cp.compute(name[i], num[i], 7, fname);
//	    		}

	    		
//	        	System.out.println("All -- "+name[i]);
//	            res.printRes();
//	            bw.write("All -- "+name[i]+"\n");
//	            bw.write(res.printResString());
	    	}
//	    	bw.close();
//		} catch (IOException e) {
			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}


    }
}
