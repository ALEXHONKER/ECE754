package cs.uwaterloo.ece.ece754;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;

import ca.uwaterloo.ece.ece754.utils.Util;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class computePvalue {
	public void compute(String projName, int num, int option,File fname){
		if(option==6){
			Util.res2csvfile(fname, compute6(num,projName));
		}else if(option==8){
			Util.res2csvfile(fname, compute8(num,projName));
		}
	}
	public String compute8(int num,String pjName){ // test
		String res=pjName+"\nP-Value,";
		for(int i=0;i<num;i++){
			BufferedReader reader9;			
			try {
				reader9 = TrainTest.getTestBufferReader(pjName, i);
				Instances testData =new Instances(reader9);
				reader9.close();
				reader9 = TrainTest.getTrainBufferReader(pjName, i);
				Instances trainData =new Instances(reader9);
				reader9.close();
				MannWhitneyUTest mtest=new MannWhitneyUTest();
				double[] p1=new double[trainData.numAttributes()];
				double[] x1=new double[trainData.size()];
				double[] x2=new double[testData.size()];
				double p=0;
				Remove remove=new Remove();
				remove.setOptions(new String[]{"-R","12-13"});
				remove.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, remove);
				remove=new Remove();
				remove.setOptions(new String[]{"-R","12-13"});
				remove.setInputFormat(testData);
				testData=Filter.useFilter(testData, remove);	
				for(int j=0;j<trainData.numAttributes();j++){
						for(int m=0;m<trainData.numInstances();m++){
							x1[m]=trainData.instance(m).value(j);
						}
						for(int m=0;m<testData.numInstances();m++){
							x2[m]=testData.instance(m).value(j);
						}
						p1[j]=mtest.mannWhitneyUTest(x1, x2);
						p+=p1[j];
				}
				//System.out.println("id:"+i);
				res+=p/testData.numAttributes();
				res+=",";
			} catch ( Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}	
			
		}
		res+="\n";
		return res;	
	}
	public String compute6(int num, String projName){ // one vs one, from back to the front
		num--;
		BufferedReader reader9;
		String res=projName+"\nP-Value,";
		try {
			reader9 = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+num+"/test.arff"));
			Instances testData9 =new Instances(reader9);
			reader9.close();
			
			for(int i=num;i>=0;i--){//from end to start, for each slot
				BufferedReader reader = new BufferedReader(
						new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+i+"/train.arff"));
				Instances trainData =new Instances(reader);
				reader.close();	
				
				List<Instance> trainList=new ArrayList<Instance>();
				//Instances trainData2 =new Instances(trainData);
				List<Instance> testList=new ArrayList<Instance>();
				
				Remove remove=new Remove();
				remove.setOptions(new String[]{"-R","12-13"});
				remove.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, remove);
				remove=new Remove();
				remove.setOptions(new String[]{"-R","12-13"});
				remove.setInputFormat(testData9);
				testData9=Filter.useFilter(testData9, remove);
				
				for(int j=0;j<trainData.numInstances();j++){
					trainList.add(trainData.instance(j));
					//trainData2.delete(0);
				}
				for(int j=0;j<testData9.numInstances();j++){
					testList.add(testData9.instance(j));
				}
				MannWhitneyUTest mtest=new MannWhitneyUTest();
				double[] p1=new double[testData9.numAttributes()];
				double[] x1=new double[trainList.size()];
				double[] x2=new double[testList.size()];
				double p=0;
				for(int j=0;j<testData9.numAttributes();j++){
						for(int m=0;m<trainData.numInstances();m++){
							x1[m]=trainData.instance(m).value(j);
						}
						for(int m=0;m<testData9.numInstances();m++){
							x2[m]=testData9.instance(m).value(j);
						}
						p1[j]=mtest.mannWhitneyUTest(x1, x2);
						p+=p1[j];
				}
				//System.out.println("id:"+i);
				res+=p/testData9.numAttributes();
				res+=",";
				System.out.println(p/testData9.numAttributes());				
			}
			res+="\n";
			
		} catch ( Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return res;
	}
}
