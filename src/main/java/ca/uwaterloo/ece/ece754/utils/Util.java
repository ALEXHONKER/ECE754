package ca.uwaterloo.ece.ece754.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Util {

	public static void arff2csv(Instances ins, String[] option, String filePath){
		Remove remove=new Remove();
		try {
			remove.setOptions(option);
			remove.setInputFormat(ins);
			Instances newTrainData=Filter.useFilter(ins, remove);
			CSVSaver cs =new  CSVSaver();
			cs.setInstances(newTrainData);
			cs.setFile(new File(filePath));
			cs.writeBatch();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	public static void printRes(double[][] matrix){
		double TN=matrix[0][0];
		double TP=matrix[1][1];
		double FP=matrix[0][1];
		double FN=matrix[1][0];
		double pre=TP/(TP+FP);
		double rec=TP/(TP+FN);
		double f1=2*pre*rec/(pre+rec);
		System.out.println("P:"+pre);
		System.out.println("R:"+rec);
		System.out.println("F1:"+f1);
	}
	public static double getPrec(double[][] matrix){
		//double TN=matrix[0][0];
		double TP=matrix[1][1];
		double FP=matrix[0][1];
		double FN=matrix[1][0];
		double pre=TP/(TP+FP);
		double rec=TP/(TP+FN);
		//double f1=2*pre*rec/(pre+rec);
		return pre;
	}
	public static double getRec(double[][] matrix){
		//double TN=matrix[0][0];
		double TP=matrix[1][1];
		double FN=matrix[1][0];
		//double pre=TP/(TP+FP);
		double rec=TP/(TP+FN);
		//double f1=2*pre*rec/(pre+rec);
		return rec;
	}
	public static double getF1(double[][] matrix){
		//double TN=matrix[0][0];
		double TP=matrix[1][1];
		double FP=matrix[0][1];
		double FN=matrix[1][0];
		double pre=TP/(TP+FP);
		double rec=TP/(TP+FN);
		double f1=2*pre*rec/(pre+rec);
		return f1;
	}
	public static void res2csvfile(File fname,String data){
		try {
			BufferedWriter bw=new BufferedWriter(new FileWriter(fname,true));
			bw.write(data);
			bw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
}
