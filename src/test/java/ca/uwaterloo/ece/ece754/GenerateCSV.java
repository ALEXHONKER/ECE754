package ca.uwaterloo.ece.ece754;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import org.junit.Test;

import ca.uwaterloo.ece.ece754.utils.Util;
import cs.uwaterloo.ece.ece754.TrainTest;
import cs.uwaterloo.ece.ece754.evalRes;
import weka.core.Instances;

public class GenerateCSV {
    @Test public void generate() {
    	String projName = "jackrabbit";
    	int id=0;
    	BufferedReader reader;
		try {
			reader = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/train.arff"));
			Instances trainData =new Instances(reader);
			reader.close();
			trainData.setClassIndex(trainData.numAttributes()-1);

			reader=new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
			Instances testData =new Instances(reader);
			reader.close();
			testData.setClassIndex(testData.numAttributes()-1);
			String[] option=new String[]{"-R","12-13"};
			String trainFilePath="data/csvData/"+projName+id+"_train.csv";
			String testFilePath="data/csvData/"+projName+id+"_test.csv";
			Util.arff2csv(trainData, trainFilePath);
			option=new String[]{"-R","12-13"};
			Util.arff2csv(testData, testFilePath);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

    }
}
