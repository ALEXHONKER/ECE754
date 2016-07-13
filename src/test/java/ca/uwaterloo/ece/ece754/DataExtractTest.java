package ca.uwaterloo.ece.ece754;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.junit.Test;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

public class DataExtractTest {
    @Test public void extract() {
    	int num=8;
    	String projName="jackrabbit";
		try {
			BufferedReader reader9 = new BufferedReader(
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
				System.out.println(p/testData9.numAttributes());				
			}
			
			
			for(int i=num;false&&i>=0;i--){// from back to start
				BufferedReader reader = new BufferedReader(
						new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+num+"/train.arff"));
				Instances trainData =new Instances(reader);
				reader.close();	
				for(int j=i;j<num;j++){
					reader = new BufferedReader(
							new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+j+"/train.arff"));
					Instances preTrainData=new Instances(reader);
					for(int k=0;k<preTrainData.numInstances();k++){
						trainData.add(preTrainData.instance(k));
					}
				}
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
				System.out.println("id:"+i);
				System.out.println(p/testData9.numAttributes());
				
			}
			for(int i=0;i<0;i++){ // from start to back
				BufferedReader reader = new BufferedReader(
						new FileReader("data/mingOri/exp-data/exp-data/jackrabbit/"+i+"/train.arff"));
				Instances trainData =new Instances(reader);
				reader.close();
				
				reader = new BufferedReader(
						new FileReader("data/mingOri/exp-data/exp-data/jackrabbit/"+i+"/test.arff"));
				Instances testData =new Instances(reader);
				reader.close();		
				for(int j=0;j<i;j++){
					reader = new BufferedReader(
							new FileReader("data/mingOri/exp-data/exp-data/jackrabbit/"+j+"/train.arff"));
					Instances preTrainData=new Instances(reader);
					for(int k=0;k<preTrainData.numInstances();k++){
						trainData.add(preTrainData.instance(k));
					}
				}
				List<Instance> trainList=new ArrayList<Instance>();
				//Instances trainData2 =new Instances(trainData);
				List<Instance> testList=new ArrayList<Instance>();
				
				Remove remove=new Remove();
				remove.setOptions(new String[]{"-R","12-13"});
				remove.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, remove);
				remove=new Remove();
				remove.setOptions(new String[]{"-R","12-13"});
				remove.setInputFormat(testData);
				testData=Filter.useFilter(testData, remove);
				
				for(int j=0;j<trainData.numInstances();j++){
					trainList.add(trainData.instance(j));
					//trainData2.delete(0);
				}
				for(int j=0;j<testData.numInstances();j++){
					testList.add(testData.instance(j));
				}
				MannWhitneyUTest mtest=new MannWhitneyUTest();
				double[] p1=new double[testData.numAttributes()];
				double[] x1=new double[trainList.size()];
				double[] x2=new double[testList.size()];
				double p=0;
				for(int j=0;j<testData.numAttributes();j++){
						for(int m=0;m<trainData.numInstances();m++){
							x1[m]=trainData.instance(m).value(j);
						}
						for(int m=0;m<testData.numInstances();m++){
							x2[m]=testData.instance(m).value(j);
						}
						p1[j]=mtest.mannWhitneyUTest(x1, x2);
						p+=p1[j];
				}
				System.out.println("id:"+i);
				System.out.println(p/testData.numAttributes());

			}

			//System.out.println(trainData.instance(2).stringValue(11));
			
			DateFormat df=new SimpleDateFormat("yyyy-MM-dd HH:mm:ssX");
			//Date dt=df.parse(trainData.instance(2).stringValue(11));
//			System.out.println(trainData.instance(1).stringValue(11));
//			System.out.println(trainData.instance(2).stringValue(11));
//			System.out.println(trainData.instance(3).stringValue(11));
//			System.out.println(trainData.instance(4).stringValue(11));
//			Reorder ro=new Reorder();
//			ro.setInputFormat(trainData);
			
			
			

			//System.out.println(trainData2.numInstances());
			
			
//			double[] x1=new double[400];
//			double[] x2=new double[400];
//			double[] x3=new double[500];
//			double[] x4=new double[500];
//			double[] x6=new double[400];
//			double[] x7=new double[400];
//			double[] x5=new double[400];
// 			int j=0;


//			for(int i=0;i<400;i++){
//				x6[i]=trainList.get(i).value(2);
//				x7[i]=trainList.get(400+i).value(2);
//				x5[i]=testList.get(i).value(2);
//			}
			
		//	double p=mtest.mannWhitneyUTest(x1, x2);
			

//			for(int i=0;i<500;i++){
//				x3[i]=testData.instance(i).value(2);
//				x4[i]=testData.instance(i).value(3);
//			}
//			double[] p1=new double[trainData.numAttributes()];
//			
//			double[] f1=new double[trainData.numInstances()];
//			double[] f2=new double[testData.numInstances()];
//			double sum=0;
//			for(int i=0;i<trainData.numAttributes();i++){
//				for(int j1=0;j1<trainData.numInstances();j1++){
//					f1[j1]=trainData.instance(j1).value(i);
//				}
//				for(int k=0;k<testData.numInstances();k++){
//					f2[k]=testData.instance(k).value(i);
//				}
//				p1[i]=mtest.mannWhitneyUTest(f1, f2);
//				sum+=p1[i];
//			}
			
			
//			System.out.println("mean:"+sum/trainData.numAttributes());
//			
//			double[] f3=new double[500];
//			double[] f4=new double[500];
//			sum=0;
//			for(int i=0;i<trainData.numAttributes();i++){
//				int ct=0;
//				for(int j1=0;j1<1000;j1+=2){
//					f3[ct]=trainData.instance(j1).value(i);
//					f4[ct++]=trainData.instance(j1+1).value(i);
//				}
//				p1[i]=mtest.mannWhitneyUTest(f3, f4);
//				sum+=p1[i];
//			}
//			System.out.println("mean2:"+sum/trainData.numAttributes());
//			
//			System.out.println("p-value1-2:"+mtest.mannWhitneyUTest(x1, x2));
//			System.out.println("p-value1-2:"+mtest.mannWhitneyUTest(x1, x3));
//			System.out.println("p-value2-3:"+mtest.mannWhitneyUTest(x2, x3));
//			System.out.println("p-value1-4:"+mtest.mannWhitneyUTest(x1, x4));
//			System.out.println("p-value6-5:"+mtest.mannWhitneyUTest(x6, x5)); // train set & test set
//			System.out.println("p-value6-7:"+mtest.mannWhitneyUTest(x6, x7));// smae train set, 400,400
//			System.out.println(ls.get(4).stringValue(11));
//			System.out.println(ls.get(5).stringValue(11));
			//List<Instance> ls=(List<Instance>) trainData;
//			trainData.sort(trainData.attribute(11));
//			Collections.sort(trainList,new Comparator<Instance>(){
//				@Override
//				public int compare(Instance o1, Instance o2) {
//					try {
//						Date dt1=df.parse(o1.stringValue(11));
//						Date dt2=df.parse(o2.stringValue(11));
//						return dt2.compareTo(dt1);
//					} catch (ParseException e) {
//						// TODO Auto-generated catch block
//						e.printStackTrace();
//					}
//					
//					return 0;
//				}
//				
//			});
//			for(Instance ins: trainList){
//				trainData2.add(ins);
//			}

//			System.out.println(trainData2.instance(1).stringValue(11));
//			System.out.println(trainData2.instance(2).stringValue(11));
//			System.out.println(trainData2.instance(3).stringValue(11));
//			System.out.println(trainData2.instance(4).stringValue(11));
//			System.out.println(ls.get(4).stringValue(11));
//			System.out.println(ls.get(5).stringValue(11));
			
			//System.out.println(trainData);
			//System.out.println(trainData.attribute(12).getDateFormat());
			//System.out.println(dt.toString());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

    }
}
