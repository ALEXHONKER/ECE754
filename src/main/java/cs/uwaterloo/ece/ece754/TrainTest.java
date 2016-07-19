package cs.uwaterloo.ece.ece754;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;

import ca.uwaterloo.ece.ece754.utils.Util;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.ADTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.AddValues;
import weka.filters.unsupervised.attribute.MergeTwoValues;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

public class TrainTest {
	public static String[] attrIndex;
	public TrainTest(){
		attrIndex=new String[]{"change_id","401_lines_added","402_lines_deleted","403_lines_changed","404_loc",
				"405_file_lines","406_time_of_day","407_day_of_week","408_previous_patches",
				"409_previous_buggy_patches","410_file_age",
				"411_commit_time","412_full_path","413_patch_lines",
				"414_lines_inserted","415_lines_removed","500_Buggy?",};
	}
	/*
	 * @param projName the name of the project
	 * @param num the number of the test&train fold
	 * @param option  0: use nothing(no resample, no feature reduction)
	 * 				  1: with SMOTE resample
	 * 				  2: with feature reduction (without SMOTE resample)
	 * 				  3: with feature reduction (with SMOTE resample)
	 * 				  7: choose 2 trianing set with the highest p-value to training the test set
	 * 				  8: original use trainx to predict testx
	 * 				  9: one nearest one with one highest p for training
	 * 				  10:  use first 16 feature for training and testing, use trainX to predict testX
	 * 				  11: use first 16 featuures, and use all previous train set for training, use Train(1~X) to predict TestX
	 * 				  12: use nearest set and another set with highest P value for training
	 * 				  13: use two training sets with highest P value for training
	 * 				  14: use one training set with highest P value for training
	 * 				  15: for each test set, use each previous training set for training
	 * 				  16: test of pvalue, (use all similar trairning set for training, divide the test set into 2 parts)
	 * 				  17: generate csv file for each training & test arff
	 * 				  18: use gloden training set for training , choose the highest training set for training
	 * @param rmOption	remove option for the training & test dataset, e.g., "-R 12-13"
	 */
	public evalRes getTestRes(String projName, int num, int option, String[] rmOption, File fname){ // option means the trianing method:
		evalRes res=new evalRes();
		
		String resCSV=projName+",\nF1,";
		if(option==0){
			for(int i=0;i<num;i++){
				double[][] tempRes=trainWithoutResample(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
			}
			return res;			
		}else if(option ==1 ){
			for(int i=0;i<num;i++){
				double[][] tempRes=trainWithSMOTEResample(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
			}
			return res;
		}else if(option==2){
			for(int i=0;i<num;i++){
				double[][] tempRes=trainWithReductionWithResample(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
			}
			return res;			
		}else if(option==4){  // use all data for trainining 
			String[][] mulOp=null;
			if(rmOption!=null){
				mulOp=new String[num][rmOption.length];
				for(int i=0;i<num;i++){
					mulOp[i]=rmOption.clone();
				}
				for(int i=0;i<num;i++){
					double[][] tempRes=trainWithSMOTEResampleAllData(projName,i,mulOp[i]);
					resCSV+=Util.getF1(tempRes);
					resCSV+=",";
					res.TN+=tempRes[0][0];
					res.TP+=tempRes[1][1];
					res.FP+=tempRes[0][1];
					res.FN+=tempRes[1][0];
					System.out.println("index:"+i);
					res.printRes();
				}
			}else{
				for(int i=0;i<num;i++){
					double[][] tempRes=trainWithSMOTEResampleAllData(projName,i,rmOption);
					resCSV+=Util.getF1(tempRes);
					resCSV+=",";
					res.TN+=tempRes[0][0];
					res.TP+=tempRes[1][1];
					res.FP+=tempRes[0][1];
					res.FN+=tempRes[1][0];
					System.out.println("index:"+i);
					res.printRes();
				}
			}

			return res;		
		}else if(option==5){ // from back to the front
			String[][] mulOp=null;
			if(rmOption!=null){
				mulOp=new String[num][rmOption.length];
				for(int i=0;i<num;i++){
					mulOp[i]=rmOption.clone();
				}
				for(int i=num-1;i>=0;i--){
					double[][] tempRes=trainWithSMOTEResampleAllDataSimilarity(projName,i,mulOp[i],num-1);
					resCSV+=Util.getF1(tempRes);
					resCSV+=",";
					res.TN+=tempRes[0][0];
					res.TP+=tempRes[1][1];
					res.FP+=tempRes[0][1];
					res.FN+=tempRes[1][0];
					System.out.println("index:"+i);
					res.printRes();
				}
			}else{
				for(int i=num-1;i>=0;i--){
					double[][] tempRes=trainWithSMOTEResampleAllDataSimilarity(projName,i,rmOption,num-1);
					resCSV+=Util.getF1(tempRes);
					resCSV+=",";
					double pre=tempRes[1][1]/(tempRes[1][1]+tempRes[0][1]);
					double rec=tempRes[1][1]/(tempRes[1][1]+tempRes[1][0]);
					System.out.println("index:"+i);
					System.out.println("pre:"+pre);
					System.out.println("rec:"+rec);
					System.out.println("f1:"+pre*rec*2/(pre+rec));
					res.TN+=tempRes[0][0];
					res.TP+=tempRes[1][1];
					res.FP+=tempRes[0][1];
					res.FN+=tempRes[1][0];
					
					//res.printRes();
				}
			}

			return res;			
		}else if(option==6){ // one vs one, the last test set with previous each train set
			String[][] mulOp=null;
			if(rmOption!=null){
				mulOp=new String[num][rmOption.length];
				for(int i=0;i<num;i++){
					mulOp[i]=rmOption.clone();
				}
				for(int i=num-1;i>=0;i--){
					double[][] tempRes=trainWithSMOTEResampleOneVsOne(projName,i,mulOp[i],num-1);
					resCSV+=Util.getF1(tempRes);
					resCSV+=",";
					res.TN+=tempRes[0][0];
					res.TP+=tempRes[1][1];
					res.FP+=tempRes[0][1];
					res.FN+=tempRes[1][0];
					System.out.println("index:"+i);
					res.printRes();
					res.reset();
				}
			}else{
				for(int i=num-1;i>=0;i--){
					double[][] tempRes=trainWithSMOTEResampleOneVsOne(projName,i,rmOption,num-1);
					resCSV+=Util.getF1(tempRes);
					resCSV+=",";
					res.TN+=tempRes[0][0];
					res.TP+=tempRes[1][1];
					res.FP+=tempRes[0][1];
					res.FN+=tempRes[1][0];
					System.out.println("index:"+i);
					res.printRes();
					res.reset();
				}
			}
			resCSV+="\n";
			Util.res2csvfile(fname, resCSV);
			return res;			
		}else if(option==7){	// choose 2 trianing set with the highest p-value to training the test set
			for(int i=num-1;i>=1;i--){
				double[][] tempRes=use2trianSetWithHighestPValue(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			Util.res2csvfile(fname, resCSV);
			return res;		
		}else if(option==8){ //use trainx to predict testx
			for(int i=num-1;i>=0;i--){
				System.out.println(i);
				double[][] tempRes=option8(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				//System.out.println("index:"+i);
				//res.printRes();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;	
		}else if(option ==9){// one neatest one with one highest p
			for(int i=num-1;i>=1;i--){
				double[][] tempRes=option9(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;					
		}else if(option == 10){ // use first 16 feature for training and testing
			for(int i=num-1;i>=0;i--){
				double[][] tempRes=option10(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;				
		}else if(option ==11){
			for(int i=num-1;i>=0;i--){
				double[][] tempRes=option11(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;				
		}else if(option ==12){
			for(int i=num-1;i>=0;i--){
				double[][] tempRes=option12(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;				
		}else if(option ==13){
			for(int i=num-1;i>0;i--){
				double[][] tempRes=option13(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;					
		}else if(option ==14){
			for(int i=num-1;i>=0;i--){
				double[][] tempRes=option14(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;					
		}else if(option ==15){
			for(int i=num-1;i>=0;i--){
				resCSV+=option15(projName,i,rmOption);
			}
			Util.res2csvfile(fname, resCSV);
			return res;					
		}else if(option==16){
			for(int i=num-1;i>=0;i--){
				double[][] tempRes=option16(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;			
		}else if(option==17){
			option17(projName,num,rmOption);
			return null;
		}else if(option ==18){
			for(int i=num-1;i>=0;i--){
				double[][] tempRes=option18(projName,i,rmOption);
				resCSV+=Util.getF1(tempRes);
				resCSV+=",";
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
				//res.reset();
			}
			resCSV+="\n";
			resCSV+="Total,"+res.getPre()*100+"%,"+res.getRec()*100+"%,"+res.getFmeasure()*100+"%,\n";
			Util.res2csvfile(fname, resCSV);
			return res;					
		}
		else {
			return null;
		}
		
	}
//	public void countAndRecord(File fname, double[][] tempRes){
//		
//	}
	public static BufferedReader getTestBufferReader(String projName, int id){
		try {
			return  new BufferedReader(
					//new FileReader("data/cleanedData/"+projName+"/arffsNoiseFilteredWOTestCases/"+id+"/test.arff"));
						new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	public static BufferedReader getTrainBufferReader(String projName, int id){
		try {
			return  new BufferedReader(
						new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/train.arff"));
					//new FileReader("data/cleanedData/"+projName+"/arffsNoiseFilteredWOTestCases/"+id+"/train.arff"));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	public double getPValueOfTwoInstances(Instances trainIns, Instances testIns){
		//double[] p1=new double[trainIns.numAttributes()-1];
		double[] x1=new double[trainIns.numInstances()];
		double[] x2=new double[testIns.numInstances()];
		double p=0;
		MannWhitneyUTest mtest=new MannWhitneyUTest();
		for(int j=0;j<trainIns.numAttributes()-1;j++){
				String attrName=trainIns.attribute(j).name();
				for(int m=0;m<trainIns.numInstances();m++){
					x1[m]=trainIns.instance(m).value(j);
				}
				Attribute attr=testIns.attribute(attrName);
				for(int m=0;m<testIns.numInstances();m++){
					x2[m]=testIns.instance(m).value(attr);
				}
				p+=mtest.mannWhitneyUTest(x1, x2);
				//p+=p1[j];
		}
		return p/(trainIns.numAttributes()-1);
	}
	/*
	 * method for remove two features
	 */
	public Instances rmIns(Instances ins, String[] rmOption){
		if(rmOption==null) return ins;
		String[] rm=rmOption.clone();
		Remove remove=new Remove();
		try {
			remove.setOptions(rm);
			remove.setInputFormat(ins);
			ins=Filter.useFilter(ins, remove);
			return  ins;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return ins;
		
	}
	/*
	 * method for resample
	 */
	public Instances resample(Instances trainData){
		SpreadSubsample sampler = new SpreadSubsample();
		String Fliteroptions="-M 1.0 -X 0.0 -S 1";
		Instances res=null;
		try {
			sampler.setOptions(weka.core.Utils.splitOptions(Fliteroptions));
			sampler.setInputFormat(trainData);
			res = Filter.useFilter(trainData, sampler);
			return res;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return res;
	}
	/*
	 * ADTtree for testing and training
	 */
	public double[][] ADTreeTrainTest(Instances trainData, Instances testData){
		ADTree classifier=new ADTree();
		String[] options =new String[]{"-B","10","-E","-3"};
		try {
			classifier.setOptions(options);
			classifier.buildClassifier(trainData);
			Evaluation eval=new Evaluation(trainData);
			eval.evaluateModel(classifier, testData);
			return eval.confusionMatrix();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
		
//		String[] rm=rmOption.clone();
//		Remove remove=new Remove();
//		remove.setOptions(rm);
//		remove.setInputFormat(trainData);
//		ins=Filter.useFilter(ins, ins);

//		FilteredClassifier fclassifier=new FilteredClassifier();
//		fclassifier.setFilter(remove);
//		fclassifier.setClassifier(classifier );
//		fclassifier.buildClassifier(trainData);
	}
	
	/*
	 * 	get a list instances with common attributes
	 * 
	 */
	public Instances[] generateSimilarIns(Instances[] ins){
		HashMap<String,Integer> se=new HashMap<String,Integer>();
		for(int i=0;i<ins[0].numAttributes();i++){
			se.put(ins[0].attribute(i).name(),1);
		}
		for(int i=1;i<ins.length;i++){
			for(int j=0;j<ins[i].numAttributes();j++){
				String tempName=ins[i].attribute(j).name();
				if(se.containsKey(tempName)){
					se.put(tempName,se.get(tempName)+1);
				}else{
					ins[i].deleteAttributeAt(j);
				}
			}
		}
		List<String> deleteStr=new ArrayList<String>();
		for(String str:se.keySet()){
			if(se.get(str)!=ins.length){
				deleteStr.add(str);
			}
		}
		//System.out.println(deleteStr.size());
		//System.out.println("se1:"+se.size());
		for(String str:deleteStr){
			se.remove(str);
		}
		//System.out.println(se.size());
		for(int i=0;i<ins.length;i++){
			//System.out.println("f:"+ins[i].numAttributes());
			int ct=0;
			for(int j=0;j<ins[i].numAttributes()-1;j++){
				if(!se.containsKey(ins[i].attribute(j).name())){
					ins[i].deleteAttributeAt(j);
					j--;
					ct++;
				}
			}
			//System.out.println("ct:"+ct+"after："+ins[i].numAttributes());
		}
		List<String> attrList=new ArrayList<String>();
		int last=ins.length-1;
		for(int i=0;i<ins[last].numAttributes();i++) attrList.add(ins[last].attribute(i).name());
		for(int i=0;i<last;i++){
			Reorder re=new Reorder();
			int[] order=new int[ins[i].numAttributes()];
			for(int j=0;j<ins[last].numAttributes();j++){
				String attrName=ins[last].attribute(j).name();
				order[j]=ins[i].attribute(attrName).index();
			}
			try {
				re.setAttributeIndicesArray(order);
				re.setInputFormat(ins[i]);
				ins[i]=Filter.useFilter(ins[i], re);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return ins;
	}
	public void option17(String projName,int id, String[] rmOption){
		for(int i=0;i<id;i++){
			try {
				BufferedReader reader;
				reader = getTrainBufferReader(projName,i);
				Instances trainData =rmIns(getFirst17Features(new Instances(reader)),rmOption);
				reader.close();
				trainData.setClassIndex(trainData.numAttributes()-1);
				reader=getTestBufferReader(projName,i);
				Instances testData;
				testData = rmIns(getFirst17Features(new Instances(reader)),rmOption);
				reader.close();
				testData.setClassIndex(testData.numAttributes()-1);
				String trainFilePath="data/csvData/"+projName+i+"_train.csv";
				String testFilePath="data/csvData/"+projName+i+"_test.csv";
				Util.arff2csv(trainData,  trainFilePath);
				Util.arff2csv(testData, testFilePath);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	public double[][] option8(String projName, int id, String[] rmOption){
		try {
			BufferedReader reader = getTestBufferReader(projName,id);
			Instances testData =new Instances(reader);
			reader.close();		
			reader =getTrainBufferReader(projName,id);
			Instances trainData=new Instances(reader);
			//testData=rmIns(testData,rmOption);
			testData.setClassIndex(testData.numAttributes()-1);
			//testData=resample(testData);
			//trainData=rmIns(trainData,rmOption);
			trainData.setClassIndex(trainData.numAttributes()-1);
			trainData=resample(trainData);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	/*
	 * get the first 17 features from a given instances, and generate a new instances
	 */
	public Instances getFirst17Features(Instances ins){
		Reorder re=new Reorder();
		int[] order=new int[17];
		for(int i=0;i<=16;i++){
			order[i]=ins.attribute(attrIndex[i]).index();
		}
		//order[16]=ins.numAttributes()-1;
		try {
			re.setAttributeIndicesArray(order);
			re.setInputFormat(ins);
			return Filter.useFilter(ins, re);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	public boolean UTest(Instances trainIns, Instances testIns){
		//double[] p1=new double[trainIns.numAttributes()-1];
		double[] x1=new double[trainIns.numInstances()];
		double[] x2=new double[testIns.numInstances()];
		double p=0;
		MannWhitneyUTest mtest=new MannWhitneyUTest();
		for(int j=0;j<trainIns.numAttributes()-1;j++){
				if(!trainIns.attribute(j).isNominal()){
					String attrName=trainIns.attribute(j).name();
					for(int m=0;m<trainIns.numInstances();m++){
						x1[m]=trainIns.instance(m).value(j);
					}
					Attribute attr=testIns.attribute(attrName);
					for(int m=0;m<testIns.numInstances();m++){
						x2[m]=testIns.instance(m).value(attr);
					}
					p=mtest.mannWhitneyUTest(x1, x2);
					System.out.println("p:"+p);
					if(p<0.00312) return false;					
				}
				//p+=p1[j];
		}
		return true;
	}
	public double[][] option16(String projName, int testId, String[] rmOption){
		try {
			Instances testData =getFirst17Features(new Instances(getTestBufferReader(projName,testId)));
			Instances test1=getFirst17Features(new Instances(getTestBufferReader(projName,testId)));
			test1.delete();
			for(int i=0;i<testData.numInstances()/2;i++) test1.add(testData.instance(i));
			Instances test2=getFirst17Features(new Instances(getTestBufferReader(projName,testId)));
			test2.delete();
			for(int i=testData.numInstances()/2;i<testData.numInstances();i++) test2.add(testData.instance(i));
			
			Instances trainData1=getFirst17Features(new Instances(getTrainBufferReader(projName,testId)));
			trainData1.delete();
			Instances trainData2=getFirst17Features(new Instances(getTrainBufferReader(projName,testId)));
			trainData2.delete();
			for(int i=0;i<=testId;i++){
				for(int m=0;m<testData.numInstances();m++){
					Instances tempTest=new Instances(testData);
					tempTest.delete();
					for(int n=m*200;m<testData.numInstances()&&n<(m+1)*200;m++){
						tempTest.add(testData.instance(n));
					}
					Instances trainTemp=getFirst17Features(new Instances(getTrainBufferReader(projName,i)));
					for(int j=0;j<trainTemp.numInstances()/200;j++){
						Instances trainss=new Instances(trainTemp);
						trainss.delete();
						for(int k=j*200;k<trainTemp.numInstances()&&k<(j+1)*200;k++){
							trainss.add(trainTemp.instance(k));
							
						}
						if(UTest(trainss,tempTest)){ 
							System.out.println("xixixi");
							trainData1=mergeIns(new Instances[]{trainData1,trainss});
						}
//						if(UTest(trainss,tempTest)){ 
//							System.out.println("xixixi");
//							trainData2=mergeIns(new Instances[]{trainData2,trainss});
//						}
					}					
				}

//				if(UTest(trainTemp,test1)){
//					System.out.println("xixixi");
//					trainData1=mergeIns(new Instances[]{trainData1,trainTemp});
//				}
//				if(UTest(trainTemp,test2)){
//					System.out.println("xixixi2");
//					trainData2=mergeIns(new Instances[]{trainData2,trainTemp});
//				}
			}
			Instances trainData=getFirst17Features(new Instances(getTrainBufferReader(projName,testId)));
			if(trainData1.numInstances()==0) {System.out.println("ffuck"); trainData1=trainData;}
			if(trainData2.numInstances()==0) {System.out.println("ffuck2"); trainData2=trainData;}
			
			trainData1.setClassIndex(trainData1.numAttributes()-1);
			trainData2.setClassIndex(trainData2.numAttributes()-1);
			test1.setClassIndex(test1.numAttributes()-1);
			test2.setClassIndex(test2.numAttributes()-1);
			trainData1=resample(trainData1);
			trainData2=resample(trainData2);
			double[][] matrix1= ADTreeTrainTest(trainData1,test1);
			double[][] matrix2= ADTreeTrainTest(trainData2,test2);
			for(int i=0;i<matrix1.length;i++){
				for(int j=0;j<matrix1[0].length;j++){
					matrix1[i][j]+=matrix2[i][j];
				}
			}
			return matrix1;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	/*
	 * generate gloden set, for each test set, use traing set with highest f1
	 */
 	public double[][] option18(String projName, int testId, String[] rmOption){
		try {
			String f1="F1,";
			String pvalue="P-value,";
			double[][] max=new double[2][2];
			double maxf1=0;
			for(int i=testId;i>=0;i--){
				BufferedReader reader = getTestBufferReader(projName,testId);
				Instances testData =new Instances(reader);
				reader.close();		
				reader =getTrainBufferReader(projName,i);
				Instances trainData=new Instances(reader);
				//testData=rmIns(testData,rmOption);
				testData=getFirst17Features(testData);
				testData.setClassIndex(testData.numAttributes()-1);
				//testData=resample(testData);
				trainData=getFirst17Features(trainData);
				//trainData=rmIns(trainData,rmOption);
				trainData.setClassIndex(trainData.numAttributes()-1);
				trainData=resample(trainData);
				double[][] matrix= ADTreeTrainTest(trainData,testData);
				f1+=Util.getF1(matrix);
				if(Util.getF1(matrix)>maxf1){
					max=matrix;
					maxf1=Util.getF1(matrix);
				}
			}
			return max;

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
 	public String option15(String projName, int testId, String[] rmOption){
		try {
			String f1="F1,";
			String pvalue="P-value,";
			for(int i=testId;i>=0;i--){
				BufferedReader reader = getTestBufferReader(projName,testId);
				Instances testData =new Instances(reader);
				reader.close();		
				reader =getTrainBufferReader(projName,i);
				Instances trainData=new Instances(reader);
				//testData=rmIns(testData,rmOption);
				testData=getFirst17Features(testData);
				testData.setClassIndex(testData.numAttributes()-1);
				//testData=resample(testData);
				trainData=getFirst17Features(trainData);
				//trainData=rmIns(trainData,rmOption);
				trainData.setClassIndex(trainData.numAttributes()-1);
				trainData=resample(trainData);
				double[][] matrix= ADTreeTrainTest(trainData,testData);
				f1+=Util.getF1(matrix);
				f1+=",";
				pvalue+=this.getPValueOfTwoInstances(rmIns(trainData,rmOption), rmIns(testData,rmOption));
				pvalue+=",";
			}
			f1+="\n";
			f1+=pvalue;
			f1+="\n";
			return f1;

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	/*
	 * get one set with highest P value for training
	 */
 	public double[][] option14(String projName, int id, String[] rmOption){
		try {
			Instances testData = getFirst17Features(new Instances(getTestBufferReader(projName,id)));
			Instances rmTestData=rmIns(getFirst17Features(new Instances(getTestBufferReader(projName,id))),rmOption);
			double highestInstances=-1;
			int idIns=0;
			//pick two train set with the highest P-value
			for(int i=0;i<=id;i++){
				Instances tempTrainData=getFirst17Features(new Instances(getTrainBufferReader(projName,i)));
				tempTrainData=rmIns(tempTrainData,rmOption);
				//Instances[] tempTrainTest= new Instances[]{tempTrainData,rmTestData};
				double p=getPValueOfTwoInstances(tempTrainData,rmTestData);
				if(p>=highestInstances){
					highestInstances=p;
					idIns=i;
				}
			}			
			Instances trainData=getFirst17Features(new Instances(getTrainBufferReader(projName,idIns)));
			//trainData=rmIns(trainData,rmOption);			
			//Instances trainData2=getFirst17Features(new Instances( getTrainBufferReader(projName,id)));
			//trainData2=rmIns(trainData2,rmOption);
			//Instances[] tempIns=new Instances[]{trainData,trainData2};
			//trainData=mergeIns(tempIns);
			
			//testData=rmIns(tempIns[2],rmOption);
			testData.setClassIndex(testData.numAttributes()-1);
			trainData.setClassIndex(trainData.numAttributes()-1);
			trainData= resample(trainData);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;	
	}	
	/*
	 * get two sets with highest P value for training
	 */
 	public double[][] option13(String projName, int id, String[] rmOption){
		try {
			Instances testData = getFirst17Features(new Instances(getTestBufferReader(projName,id)));
			Instances rmTestData=rmIns(getFirst17Features(new Instances(getTestBufferReader(projName,id))),rmOption);
			double[] highestInstances=new double[]{-1,-1};
			int[] idIns=new int[]{-1,-1};
			//pick two train set with the highest P-value
			for(int i=0;i<=id;i++){
				Instances tempTrainData=getFirst17Features(new Instances(getTrainBufferReader(projName,i)));
				tempTrainData=rmIns(tempTrainData,rmOption);
				//Instances[] tempTrainTest= new Instances[]{tempTrainData,rmTestData};
				double p=getPValueOfTwoInstances(tempTrainData,rmTestData);
				if(p>=Math.min(highestInstances[0],highestInstances[1])){
					if(highestInstances[0]>=highestInstances[1]){
						highestInstances[1]=p;
						idIns[1]=i;
					}else{
						highestInstances[0]=p;
						idIns[0]=i;
					}
				}
			}			
			Instances trainData=getFirst17Features(new Instances(getTrainBufferReader(projName,idIns[0])));
			//trainData=rmIns(trainData,rmOption);			
			Instances trainData2=getFirst17Features(new Instances( getTrainBufferReader(projName,idIns[1])));
			//trainData2=rmIns(trainData2,rmOption);
			Instances[] tempIns=new Instances[]{trainData,trainData2};
			trainData=mergeIns(tempIns);
			
			//testData=rmIns(tempIns[2],rmOption);
			testData.setClassIndex(testData.numAttributes()-1);
			trainData.setClassIndex(trainData.numAttributes()-1);
			trainData= resample(trainData);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;	
	}
 	public double[][] option12(String projName, int id, String[] rmOption){
		try {
			Instances testData = getFirst17Features(new Instances(getTestBufferReader(projName,id)));
			Instances rmTestData=rmIns(getFirst17Features(new Instances(getTestBufferReader(projName,id))),rmOption);
			double highestInstances=-1;
			int idIns=0;
			//pick two train set with the highest P-value
			for(int i=0;i<id;i++){
				Instances tempTrainData=getFirst17Features(new Instances(getTrainBufferReader(projName,i)));
				tempTrainData=rmIns(tempTrainData,rmOption);
				//Instances[] tempTrainTest= new Instances[]{tempTrainData,rmTestData};
				double p=getPValueOfTwoInstances(tempTrainData,rmTestData);
				if(p>=highestInstances){
					highestInstances=p;
					idIns=i;
				}
			}			
			Instances trainData=getFirst17Features(new Instances(getTrainBufferReader(projName,idIns)));
			//trainData=rmIns(trainData,rmOption);			
			Instances trainData2=getFirst17Features(new Instances( getTrainBufferReader(projName,id)));
			//trainData2=rmIns(trainData2,rmOption);
			Instances[] tempIns=new Instances[]{trainData,trainData2};
			trainData=mergeIns(tempIns);
			
			//testData=rmIns(tempIns[2],rmOption);
			testData.setClassIndex(testData.numAttributes()-1);
			trainData.setClassIndex(trainData.numAttributes()-1);
			trainData= resample(trainData);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;	
	}
 	/*
 	 * method for merging mulitple instances
 	 * @param trainData the array of instances, each instances should only contains 17 features;
 	 */
 	public Instances mergeIns(Instances[] trainData){
		Set<String> setAttr11=new HashSet<String>();
		Enumeration<Object> ens=trainData[0].attribute("411_commit_time").enumerateValues();
		while(ens.hasMoreElements()){
			setAttr11.add((String)ens.nextElement());
		}	
		Set<String> setAttr12=new HashSet<String>();
		ens=trainData[0].attribute("412_full_path").enumerateValues();
		while(ens.hasMoreElements()){
			setAttr12.add((String)ens.nextElement());
		}
		Instances res=trainData[0];
		Instances tempIns;
		for(int i=1;i<trainData.length;i++){
			tempIns=trainData[i];
			AddValues av=new AddValues();
			Enumeration<Object> en=tempIns.attribute("411_commit_time").enumerateValues();
			StringBuilder sb=new StringBuilder();
			while(en.hasMoreElements()){
				String str=(String)en.nextElement();
				if(!setAttr11.contains(str)){
					setAttr11.add(str);
					sb.append(str+",");
				}
			}
			try {		
				if(sb.length()>0) 
					sb.delete(sb.length()-1, sb.length());
				av.setAttributeIndex((res.attribute("411_commit_time").index()+1)+"");
				av.setLabels(sb.toString());
				av.setInputFormat(res);
				res=Filter.useFilter(res, av);
				
				sb=new StringBuilder();
				en=tempIns.attribute("412_full_path").enumerateValues();
				while(en.hasMoreElements()){
					String str=(String)en.nextElement();
					if(!setAttr12.contains(str)){
						setAttr12.add(str);
						sb.append(str+",");
					}
				}
				if(sb.length()>0)  sb.delete(sb.length()-1, sb.length());
				av.setAttributeIndex((res.attribute("412_full_path").index()+1)+"");
				av.setLabels(sb.toString());
				av.setInputFormat(res);
				res=Filter.useFilter(res, av);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			res.addAll(tempIns);
			//trainData = Util.merge(trainData,tempIns);
		}
		return res;
 	}
 	public double[][] option11(String projName, int id, String[] rmOption){
		try {
			BufferedReader reader = getTestBufferReader(projName,id);
			Instances testData =new Instances(reader);
			reader.close();		
			reader =getTrainBufferReader(projName,id);
			Instances trainData=new Instances(reader);
			//testData=rmIns(testData,rmOption);
			testData=getFirst17Features(testData);
			testData.setClassIndex(testData.numAttributes()-1);
			trainData=getFirst17Features(trainData);
			//trainData.setClassIndex(trainData.numAttributes()-1);
			
			Instances[] insArray=new Instances[id+1];
			insArray[0]=trainData;
			for(int i=0;i<id;i++){
				insArray[i+1]=getFirst17Features(new Instances(getTrainBufferReader(projName,i)));
			}
			trainData=mergeIns(insArray);
			
//			Set<String> setAttr11=new HashSet<String>();
//			Enumeration<Object> ens=trainData.attribute("411_commit_time").enumerateValues();
//			while(ens.hasMoreElements()){
//				setAttr11.add((String)ens.nextElement());
//			}		
//			Set<String> setAttr12=new HashSet<String>();
//			ens=trainData.attribute("412_full_path").enumerateValues();
//			while(ens.hasMoreElements()){
//				setAttr12.add((String)ens.nextElement());
//			}	
//			for(int i=0;i<id;i++){
//				Instances tempIns=new Instances(getTrainBufferReader(projName,i));
//				tempIns=getFirst17Features(tempIns);
//				AddValues av=new AddValues();
//				Enumeration<Object> en=tempIns.attribute(11).enumerateValues();
//				StringBuilder sb=new StringBuilder();
//				while(en.hasMoreElements()){
//					String str=(String)en.nextElement();
//					if(!setAttr11.contains(str)){
//						setAttr11.add(str);
//						sb.append(str+",");
//					}
//				}
//				if(sb.length()>0)sb.delete(sb.length()-1, sb.length());
//				av.setAttributeIndex((trainData.attribute("411_commit_time").index()+1)+"");
//				av.setLabels(sb.toString());
//				av.setInputFormat(trainData);
//				trainData=Filter.useFilter(trainData, av);
//				
//				sb=new StringBuilder();
//				en=tempIns.attribute(12).enumerateValues();
//				while(en.hasMoreElements()){
//					String str=(String)en.nextElement();
//					if(!setAttr12.contains(str)){
//						setAttr12.add(str);
//						sb.append(str+",");
//					}
//				}
//				if(sb.length()>0) sb.delete(sb.length()-1, sb.length());
//				av.setAttributeIndex((trainData.attribute("412_full_path").index()+1)+"");
//				av.setLabels(sb.toString());
//				av.setInputFormat(trainData);
//				trainData=Filter.useFilter(trainData, av);
//				trainData.addAll(tempIns);
//				//trainData = Util.merge(trainData,tempIns);
//			}
//			//trainData=rmIns(trainData,rmOption);
			trainData.setClassIndex(trainData.numAttributes()-1);
			trainData=resample(trainData);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}	
 	public double[][] option10(String projName, int id, String[] rmOption){
		try {
			BufferedReader reader = getTestBufferReader(projName,id);
			Instances testData =new Instances(reader);
			reader.close();		
			reader =getTrainBufferReader(projName,id);
			Instances trainData=new Instances(reader);
			//testData=rmIns(testData,rmOption);
			testData=getFirst17Features(testData);
			testData.setClassIndex(testData.numAttributes()-1);
			//testData=resample(testData);
			trainData=getFirst17Features(trainData);
			//trainData=rmIns(trainData,rmOption);
			trainData.setClassIndex(trainData.numAttributes()-1);
			trainData=resample(trainData);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
 	
 	public double[][] option9(String projName, int id, String[] rmOption){
		BufferedReader reader;	
		try {
			reader =getTestBufferReader(projName,id); 
//					new BufferedReader(
//					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
			Instances testData =new Instances(reader);
			reader.close();
			System.out.println("testlabel:"+testData.instance(0).value(testData.numAttributes()-1));
			reader= getTestBufferReader(projName,id); 
//			reader = new BufferedReader(
//					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
			Instances rmTestData=rmIns(new Instances(reader),rmOption);
			reader.close();
			double highestInstances=-1;
			int idIns=0;
			//pick two train set with the highest P-value
			for(int i=0;i<id;i++){
				reader = getTrainBufferReader(projName,i);
				Instances tempTrainData=new Instances(reader);
				tempTrainData=rmIns(tempTrainData,rmOption);
				Instances[] tempTrainTest= generateSimilarIns(new Instances[]{tempTrainData,rmTestData});
				System.out.println(tempTrainTest[0].equalHeaders(tempTrainTest[1]));
				reader.close();
				double p=getPValueOfTwoInstances(tempTrainTest[0],tempTrainTest[1]);
				if(p>=highestInstances){
					highestInstances=p;
					idIns=i;
				}
			}			
			reader = getTrainBufferReader(projName,idIns);
			Instances trainData=new Instances(reader);
			trainData=rmIns(trainData,rmOption);			
			reader = getTrainBufferReader(projName,id);	
			Instances trainData2=new Instances(reader);
			trainData2=rmIns(trainData2,rmOption);
			Instances[] tempIns=generateSimilarIns(new Instances[]{trainData,trainData2,testData});
			reader.close();
			tempIns[0].addAll(tempIns[1]);
			trainData=tempIns[0];
			//testData=tempIns[2];
			testData=rmIns(tempIns[2],rmOption);
			testData.setClassIndex(testData.numAttributes()-1);
			trainData.setClassIndex(trainData.numAttributes()-1);
			trainData= resample(trainData);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;	
	}
 	public double[][] use2trianSetWithHighestPValue(String projName, int id, String[] rmOption){
		BufferedReader reader;	
		try {
			reader = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
			Instances testData =new Instances(reader);
			reader.close();
			System.out.println("testlabel:"+testData.instance(0).value(testData.numAttributes()-1));
			reader = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
			Instances rmTestData=rmIns(new Instances(reader),rmOption);
			reader.close();
			double[] highestInstances=new double[]{-1,-1};
			int[] idIns=new int[]{-1,-1};
			//pick two train set with the highest P-value
			for(int i=0;i<=id;i++){
				reader = getTrainBufferReader(projName,i);
				Instances tempTrainData=new Instances(reader);
				tempTrainData=rmIns(tempTrainData,rmOption);
				Instances[] tempTrainTest= generateSimilarIns(new Instances[]{tempTrainData,rmTestData});
				System.out.println(tempTrainTest[0].equalHeaders(tempTrainTest[1]));
				reader.close();
				double p=getPValueOfTwoInstances(tempTrainTest[0],tempTrainTest[1]);
				
				if(p>Math.min(highestInstances[0],highestInstances[1])){
					if(highestInstances[0]<=highestInstances[1]){
						highestInstances[0]=p;
						idIns[0]=i;
					}else{
						highestInstances[1]=p;
						idIns[1]=i;
					}
				}
			}
			
			reader = getTrainBufferReader(projName,idIns[0]);
			Instances trainData=new Instances(reader);
			trainData=rmIns(trainData,rmOption);
			
			reader = getTrainBufferReader(projName,idIns[1]);	
			Instances trainData2=new Instances(reader);
			trainData2=rmIns(trainData2,rmOption);
//			System.out.println(trainData.equalHeaders(testData));
//			System.out.println("nominal?"+trainData.attribute(trainData.numAttributes()-1).isNominal());
			Instances[] tempIns=generateSimilarIns(new Instances[]{trainData,trainData2,testData});
//			System.out.println("train1_2:"+trainData.equalHeaders(trainData2));
			reader.close();

			//trainData.addAll(trainData2);
//			System.out.println("equal1:"+tempIns[0].equalHeaders(tempIns[1]));
//			System.out.println("equal2:"+tempIns[0].equalHeaders(tempIns[2]));
//			System.out.println("nominal?"+tempIns[0].attribute(tempIns[0].numAttributes()-1).isNominal());
			tempIns[0].addAll(tempIns[1]);
			trainData=tempIns[0];
			//System.out.println(trainData.equalHeaders(testData));
			testData=rmIns(tempIns[2],rmOption);
//			System.out.println(trainData.equalHeaders(testData));
//			System.out.println("NumTes:"+testData.numInstances());
//			System.out.println("NumTesAttr:"+testData.numAttributes());
//			trainData=rmIns(trainData,rmOption);
			
//			System.out.println(trainData.equalHeaders(trainData2));
//			System.out.println(trainData.equalHeaders(testData));
			testData.setClassIndex(testData.numAttributes()-1);
			trainData.setClassIndex(trainData.numAttributes()-1);
			//System.out.println("label:"+trainData.instance(0).value(trainData.numAttributes()-1));
//			Resample sv= new Resample();
//			sv.setInputFormat(trainData);
//			trainData=Filter.useFilter(trainData, sv);
//			SMOTE sm=new SMOTE();
//			sm.setInputFormat(trainData);
//			trainData=Filter.useFilter(trainData, sm);
//			System.out.println("numIns:"+trainData.numInstances());
//			System.out.println("numFea:"+trainData.numAttributes());
//			System.out.println("nominal2?"+trainData.attribute(trainData.numAttributes()-1).isNominal());
			trainData= resample(trainData);
//			System.out.println("TrainNumIns:"+trainData.numInstances());
//			System.out.println("TrainSize:"+trainData.size());
//			System.out.println("TestSize:"+testData.size());
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
		
		
	}
	public double[][] trainWithSMOTEResampleOneVsOne(String projName, int id, String[] rmOption,int sum){	
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/train.arff"));
			Instances trainData =new Instances(reader);
			reader.close();
			String[][] op=null;
			if(rmOption!=null){
				op=new String[id+1][rmOption.length];
				for(int i=0;i<id+1;i++){
					op[i]=rmOption.clone();
				}				
			}
			
			if(rmOption!=null){
				Remove remove=new Remove();
				remove.setOptions(rmOption);
				remove.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, remove);
			}
			
//			System.out.println("num:"+trainData.numInstances());
//			System.out.println("attr:"+trainData.numAttributes());
			trainData.setClassIndex(trainData.numAttributes()-1);

			reader=new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+sum+"/test.arff"));
			Instances testData =new Instances(reader);
			reader.close();
			if(rmOption!=null){
				Remove remove=new Remove();
				remove.setOptions(op[id]);
				remove.setInputFormat(testData);
				testData=Filter.useFilter(testData, remove);
			}
			testData.setClassIndex(testData.numAttributes()-1);
			SMOTE sm=new SMOTE();
			sm.setInputFormat(trainData);
			trainData=Filter.useFilter(trainData, sm);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	public double[][] trainWithSMOTEResampleAllDataSimilarity(String projName, int id, String[] rmOption,int sum){	
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+sum+"/train.arff"));
			Instances trainData =new Instances(reader);
			reader.close();
			String[][] op=null;
			if(rmOption!=null){
				op=new String[id+1][rmOption.length];
				for(int i=0;i<id+1;i++){
					op[i]=rmOption.clone();
				}				
			}
			
			if(rmOption!=null){
				Remove remove=new Remove();
				remove.setOptions(rmOption);
				remove.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, remove);
			}
			System.out.println("num:"+trainData.numInstances());
			System.out.println("attr:"+trainData.numAttributes());
			for(int i=id;i<sum;i++){
				reader = new BufferedReader(
						new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+i+"/train.arff"));
				Instances preTrainData=new Instances(reader);
				if(rmOption!=null){
					Remove remove=new Remove();
					remove.setOptions(op[i]);
					remove.setInputFormat(preTrainData);
					preTrainData=Filter.useFilter(preTrainData, remove);
				}
				//trainData=Instances.mergeInstances(trainData, preTrainData);
				for(int j=0;i<preTrainData.numInstances();i++){
					trainData.add(preTrainData.instance(j));
				}
			}
			
//			System.out.println("num:"+trainData.numInstances());
//			System.out.println("attr:"+trainData.numAttributes());
			trainData.setClassIndex(trainData.numAttributes()-1);

			reader=new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+sum+"/test.arff"));
			Instances testData =new Instances(reader);
			reader.close();
			if(rmOption!=null){
				Remove remove=new Remove();
				remove.setOptions(op[id]);
				remove.setInputFormat(testData);
				testData=Filter.useFilter(testData, remove);
			}
			
			MannWhitneyUTest mTest=new MannWhitneyUTest();
//			
//			System.out.println("TESTnum:"+testData.numInstances());
//			System.out.println("TESTattr:"+testData.numAttributes());
			testData.setClassIndex(testData.numAttributes()-1);
			SMOTE sm=new SMOTE();
			sm.setInputFormat(trainData);
			trainData=Filter.useFilter(trainData, sm);
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	public double[][] trainWithSMOTEResampleAllData(String projName, int id, String[] rmOption){	
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/train.arff"));
			Instances trainData =new Instances(reader);
			reader.close();
			String[][] op=null;
			if(rmOption!=null){
				op=new String[id+1][rmOption.length];
				for(int i=0;i<id+1;i++){
					op[i]=rmOption.clone();
				}				
			}
			
			if(rmOption!=null){
				Remove remove=new Remove();
				remove.setOptions(rmOption);
				remove.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, remove);
			}
			System.out.println("num:"+trainData.numInstances());
			System.out.println("attr:"+trainData.numAttributes());
			for(int i=0;i<id;i++){
				reader = new BufferedReader(
						new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+i+"/train.arff"));
				Instances preTrainData=new Instances(reader);
				if(rmOption!=null){
					Remove remove=new Remove();
					remove.setOptions(op[i]);
					remove.setInputFormat(preTrainData);
					preTrainData=Filter.useFilter(preTrainData, remove);
				}
				//trainData=Instances.mergeInstances(trainData, preTrainData);
				for(int j=0;i<preTrainData.numInstances();i++){
					trainData.add(preTrainData.instance(j));
				}
			}
			
			System.out.println("num:"+trainData.numInstances());
			System.out.println("attr:"+trainData.numAttributes());
			trainData.setClassIndex(trainData.numAttributes()-1);

			reader=new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
			Instances testData =new Instances(reader);
			reader.close();
			if(rmOption!=null){
				Remove remove=new Remove();
				remove.setOptions(op[id]);
				remove.setInputFormat(testData);
				testData=Filter.useFilter(testData, remove);
			}
			System.out.println("TESTnum:"+testData.numInstances());
			System.out.println("TESTattr:"+testData.numAttributes());
			testData.setClassIndex(testData.numAttributes()-1);
			SMOTE sm=new SMOTE();
			sm.setInputFormat(trainData);
			trainData=Filter.useFilter(trainData, sm);

			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	public double[][] trainWithReductionWithResample(String projName, int id, String[] rmOption){

		BufferedReader reader;
		
		try {
			reader = new BufferedReader(
						new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/train.arff"));
			Instances trainData =new Instances(reader);
			reader.close();
			trainData.setClassIndex(trainData.numAttributes()-1);
			Attribute attr=trainData.attribute(12);
			
			
			reader=new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
			Instances testData =new Instances(reader);
			reader.close();
			testData.setClassIndex(testData.numAttributes()-1);
			
			if(rmOption!=null){
				String[] rmOption2=rmOption.clone();
				Remove remove=new Remove();
				remove.setOptions(rmOption);
				remove.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, remove);
				remove=new Remove();
				remove.setOptions(rmOption2);
				remove.setInputFormat(testData);
				testData=Filter.useFilter(testData, remove);
			}
			

			SMOTE sm=new SMOTE();
			sm.setInputFormat(trainData);
			trainData=Filter.useFilter(trainData, sm);

			PrincipalComponents PCA= new PrincipalComponents();
			AttributeSelection selector=new AttributeSelection();
			Ranker ranker=new Ranker();
			selector.setEvaluator(PCA);
			selector.setSearch(ranker);
			selector.SelectAttributes(trainData);

			trainData=selector.reduceDimensionality(trainData);
			testData=selector.reduceDimensionality(testData);
			
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		return null;
	}
	
	public double[][] trainWithoutResample(String projName, int id,String[] rmOption){
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
			if(rmOption!=null){
				Remove remove=new Remove();
				remove.setOptions(rmOption);
				remove.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, remove);
				remove=new Remove();
				remove.setOptions(rmOption);
				remove.setInputFormat(testData);
				testData=Filter.useFilter(testData, remove);
			}
			return ADTreeTrainTest(trainData,testData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	public double[][] trainWithSMOTEResample(String projName, int id, String[] rmOption){
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
				SMOTE sm=new SMOTE();
				sm.setInputFormat(trainData);
				trainData=Filter.useFilter(trainData, sm);
				if(rmOption!=null){
					Remove remove=new Remove();
					remove.setOptions(rmOption);
					remove.setInputFormat(trainData);
					trainData=Filter.useFilter(trainData, remove);
					remove=new Remove();
					remove.setOptions(rmOption);
					remove.setInputFormat(testData);
					testData=Filter.useFilter(testData, remove);
				}

				return ADTreeTrainTest(trainData,testData);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return null;
        	

	}
	
}
