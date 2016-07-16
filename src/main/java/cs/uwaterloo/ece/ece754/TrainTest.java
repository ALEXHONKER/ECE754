package cs.uwaterloo.ece.ece754;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
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
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

public class TrainTest {
	/*
	 * @param projName the name of the project
	 * @param num the number of the test&train fold
	 * @param option  0: use nothing(no resample, no feature reduction)
	 * 				  1: with SMOTE resample
	 * 				  2: with feature reduction (without SMOTE resample)
	 * 				  3: with feature reduction (with SMOTE resample)
	 * 				  7: choose 2 trianing set with the highest p-value to training the test set
	 * 				  8: original use trainx to predict testx
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
			for(int i=0;i<num;i++){
				System.out.println(i);
				double[][] tempRes=useLatestSetToTraining(projName,i,rmOption);
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
			resCSV+="Total,"+res.getFmeasure()+",\n";
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
			resCSV+="Total,"+res.getFmeasure()+",\n";
			Util.res2csvfile(fname, resCSV);
			return res;					
		}else if(option == 10){ // use first 16 feature for training and testing
			for(int i=num-1;i>=1;i--){
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
			resCSV+="Total,"+res.getFmeasure()+",\n";
			Util.res2csvfile(fname, resCSV);
			return res;				
		}
		else {
			return null;
		}
		
	}

	public static BufferedReader getTestBufferReader(String projName, int id){
		try {
			return  new BufferedReader(
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
//		Arrays.sort(ins,new Comparator<Instances>(){
//			@Override
//			public int compare(Instances o1, Instances o2) {
//				// TODO Auto-generated method stub
//				return o1.numAttributes()-o2.numAttributes();
//			}
//			
//		});
		//ins[0].attribute(0).
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
//		for(int i=0;i<last;i++){
//			Instances tempIns= new Instances(ins[last]);
//			tempIns.delete();
//			res[i]=tempIns;
//			for(int j=0;j<ins[i].numInstances();j++){
//				double[] ss1=ins[i].instance(j).toDoubleArray();
//				double[] ss2=ins[i].instance(j).toDoubleArray();
//				for(int k=0;k<ins[last].numAttributes();k++){
//					String attrName=ins[last].attribute(k).name();
//					int index=ins[i].attribute(attrName).index();
//					ss1[k]=ss2[index];
//				}
//				tempIns.add(new DenseInstance(1.0, ss1));
//			}
//		}
//		res[last]=ins[last];
//		Attribute attr=ins[0].attribute(0);
//		System.out.println("attrValue:"+ins[0].instance(10).value(0));
//		Instance inst=ins[0].instance(0);
		return ins;
	}
	public double[][] useLatestSetToTraining(String projName, int id, String[] rmOption){
		try {
			BufferedReader reader = getTestBufferReader(projName,id);
			Instances testData =new Instances(reader);
			reader.close();		
			reader =getTrainBufferReader(projName,id);
			Instances trainData=new Instances(reader);
			testData=rmIns(testData,rmOption);
			testData.setClassIndex(testData.numAttributes()-1);
			testData=resample(testData);
//			System.out.println("attrName:"+testData.attribute(2).name());
//			System.out.println("attrName2:"+testData.attribute(3).name());
			trainData=rmIns(trainData,rmOption);
			trainData.setClassIndex(trainData.numAttributes()-1);
			trainData=resample(trainData);
			Set<Attribute> se=new HashSet<Attribute>();
			se.add(testData.attribute(1));
//			System.out.println(se.contains(trainData.attribute(1)));
//			System.out.println(se.contains(trainData.attribute(2)));
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
			Reorder re=new Reorder();
			int[] order=new int[17];
			for(int i=0;i<16;i++){
				order[i]=i;
			}
			order[16]=testData.numAttributes()-1;
			re.setInputFormat(testData);
			testData=Filter.useFilter(testData, re);
			testData.setClassIndex(testData.numAttributes()-1);
			testData=resample(testData);
			for(int i=0;i<16;i++){
				order[i]=i;
			}
			order[16]=trainData.numAttributes()-1;
			re.setInputFormat(trainData);
			trainData=Filter.useFilter(trainData, re);
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
			reader = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
			Instances testData =new Instances(reader);
			reader.close();
			System.out.println("testlabel:"+testData.instance(0).value(testData.numAttributes()-1));
			reader = new BufferedReader(
					new FileReader("data/mingOri/exp-data/exp-data/"+projName+"/"+id+"/test.arff"));
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
