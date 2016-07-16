package ca.uwaterloo.ece.ece754.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Util {
	public static Instances merge(Instances data1, Instances data2)
		    throws Exception
		{
		    // Check where are the string attributes
		    int asize = data1.numAttributes();
		    boolean strings_pos[] = new boolean[asize];
		    for(int i=0; i<asize; i++)
		    {
		        Attribute att = data1.attribute(i);
		        strings_pos[i] = ((att.type() == Attribute.STRING) ||
		                          (att.type() == Attribute.NOMINAL));
		    }

		    // Create a new dataset
		    Instances dest = new Instances(data1);
		    dest.setRelationName(data1.relationName() + "+" + data2.relationName());

		    DataSource source = new DataSource(data2);
		    Instances instances = source.getStructure();
		    Instance instance = null;
		    while (source.hasMoreElements(instances)) {
		        instance = source.nextElement(instances);
		        dest.add(instance);

		        // Copy string attributes
		        for(int i=0; i<asize; i++) {
		            if(strings_pos[i]) {
		                dest.instance(dest.numInstances()-1)
		                    .setValue(i,instance.stringValue(i));
		            }
		        }
		    }

		    return dest;
		}
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
