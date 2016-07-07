package ca.uwaterloo.ece.ece754.utils;

import java.io.File;

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
}
