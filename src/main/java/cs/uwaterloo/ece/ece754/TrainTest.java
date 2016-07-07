package cs.uwaterloo.ece.ece754;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

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
	public evalRes getTestRes(String projName, int num, int option, String[] rmOption){ // option means the trianing method:
		evalRes res=new evalRes();
		if(option==0){
			for(int i=0;i<num;i++){
				double[][] tempRes=trainWithoutResample(projName,i,rmOption);
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
				res.TN+=tempRes[0][0];
				res.TP+=tempRes[1][1];
				res.FP+=tempRes[0][1];
				res.FN+=tempRes[1][0];
				System.out.println("index:"+i);
				res.printRes();
			}
			return res;			
		}else {
			return null;
		}
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
			
//			PrincipalComponents PCA= new PrincipalComponents();
//			PCA.setInputFormat(trainData);
//			PCA.setMaximumAttributes(500);
			
//			System.out.println(trainData.numAttributes());
//			trainData=Filter.useFilter(trainData,PCA);
//			System.out.println(trainData.numAttributes());
//			//testData=Filter.useFilter(testData,PCA);
//			System.out.println(trainData.numAttributes());

//			AttributeSelection as=new AttributeSelection();
//			as.reduceDimensionality(trainData);
//			System.out.println("ohehe:");

			

//			AttributeSelectedClassifier asc=new AttributeSelectedClassifier();
//			asc.setClassifier(classifier);
//			asc.buildClassifier(trainData);
//			ASEvaluation ase=asc.getEvaluator();
//			ase.buildEvaluator(trainData);
			// resampling
			//System.out.println("trains:"+trainData.numInstances());
			SMOTE sm=new SMOTE();
			sm.setInputFormat(trainData);
			trainData=Filter.useFilter(trainData, sm);
			//System.out.println("trains:"+trainData.numInstances());
			/////////
			PrincipalComponents PCA= new PrincipalComponents();
			AttributeSelection selector=new AttributeSelection();
			Ranker ranker=new Ranker();
			selector.setEvaluator(PCA);
			selector.setSearch(ranker);
			selector.SelectAttributes(trainData);
			//System.out.println("train:"+trainData.numAttributes());
			//System.out.println("test:"+testData.numAttributes());
			trainData=selector.reduceDimensionality(trainData);
			testData=selector.reduceDimensionality(testData);
			//System.out.println("train:"+trainData.numAttributes());
			//System.out.println("test:"+testData.numAttributes());

			
			Classifier classifier=new ADTree();
			String[] options =new String[]{"-B","10","-E","-3"};
			classifier.setOptions(options);
			classifier.buildClassifier(trainData);
			//////
//			FilteredClassifier fc=new FilteredClassifier();
//			fc.setFilter(PCA);
//			fc.setClassifier(classifier);
			
			
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
