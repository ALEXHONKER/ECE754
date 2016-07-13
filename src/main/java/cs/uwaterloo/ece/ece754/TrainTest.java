package cs.uwaterloo.ece.ece754;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

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
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Remove;

public class TrainTest {
	/*
	 * @param projName the name of the project
	 * @param num the number of the test&train fold
	 * @param option  0: use nothing(no resample, no feature reduction)
	 * 				  1: with SMOTE resample
	 * 				  2: with feature reduction (without SMOTE resample)
	 * 				  3: with feature reduction (with SMOTE resample)
	 * @param rmOption	remove option for the training & test dataset, e.g., "-R 12-13"
	 */
	public evalRes getTestRes(String projName, int num, int option, String[] rmOption, File fname){ // option means the trianing method:
		evalRes res=new evalRes();
		
		String resCSV="F1,";
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
		} 
		else {
			return null;
		}
		
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

			Classifier classifier=new ADTree();
			String[] options =new String[]{"-B","10","-E","-3"};
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

			Classifier classifier=new ADTree();
			String[] options =new String[]{"-B","10","-E","-3"};
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

			Classifier classifier=new ADTree();
			String[] options =new String[]{"-B","10","-E","-3"};
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


			
			Classifier classifier=new ADTree();
			String[] options =new String[]{"-B","10","-E","-3"};
			classifier.setOptions(options);
			classifier.buildClassifier(trainData);
			//////
			
			
			Evaluation eval=new Evaluation(trainData);
			eval.evaluateModel(classifier, testData);
			return eval.confusionMatrix();
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
			Classifier classifier=new ADTree();
			String[] options =new String[]{"-B","10","-E","-3"};
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

				
				Classifier classifier=new ADTree();
				String[] options =new String[]{"-B","10","-E","-3"};
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
        	

	}
	
}
