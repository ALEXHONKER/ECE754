package cs.uwaterloo.ece.ece754;

public class evalRes {
	private double pre; //pre,rec,f1
	private double rec;
	private double fmeasure;
	double TP,TN,FP,FN;
	double auc;
	double[][] confMatrix;
	public evalRes(){
		this.confMatrix=new double[2][2];
		this.pre=0;
		this.rec=0;
		this.fmeasure=0;
		this.TP=0;
		this.FP=0;
		this.FN=0;
		this.TN=0;
	}
	public double getPre(){
		pre=TP/(TP+FP);
		return TP/(TP+FP);
	}
	public double getRec(){
		rec=TP/(TP+FN);
		return TP/(TP+FN);
	}
	public double getFmeasure(){
		getPre();
		getRec();
		fmeasure=2*pre*rec/(pre+rec);
		return fmeasure;
	}
	public void printRes(){
		this.getFmeasure();
		System.out.println("P:"+pre);
		System.out.println("R:"+rec);
		System.out.println("F1:"+fmeasure);
	}
	public void reset(){
		this.pre=0;
		this.rec=0;
		this.fmeasure=0;
		this.TP=0;
		this.FP=0;
		this.FN=0;
		this.TN=0;
	}
	public String printResString(){
		return "P:"+pre+"\n"+"R:"+rec+"\n"+"F1:"+fmeasure+"\n";
	}
	
}
