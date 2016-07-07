function [sm]= computeSM(ins,Wei,K, U, V,Q)
%% function for computing SM, V-->SIGMA, U-->MU
    N          = size(ins,1);      % n = Number of Pixels
    H          = zeros(N,K);
    for k      = 1:K
    h          = RBFKernel(ins,U',V,N,k);
    H(:,k)     = h;
    end
    Y          = H*Wei; 
    n=size(ins,2);
    left=(sqrt(pi)/(2*sqrt(2)*Q))^n;
    M=size(Wei,1);   %m==k the number of neuron
    I1=zeros(size(ins,1),1);
    for j=1:M
        mul=ones(size(ins,1),1);
        for i=1:n
            mul=mul.*V(j).*(erf((ins(:,i)-U(j,i)+Q)/(sqrt(2)*V(j)))-erf((ins(:,i)-U(j,i)-Q)/(sqrt(2)*V(j))));
        end
        I1=I1+mul.*Wei(j);
    end
    I1=I1.*left;
    
    left2=(sqrt(pi)/4*Q)^n;
    I2=zeros(size(ins,1),1);
    for j=1:M
        for k=1:M
            temp=sqrt(2*V(j)^2*V(k)^2*(V(k)^2+V(j)^2))^n;
            mul=ones(size(ins,1),1);
            for i=1:n
                e1=erf(((V(k)^2+V(j)^2).*(ins(:,i)+Q)-(V(k)^2*U(j,i)+V(j)^2*U(k,i)))...
                    /(sqrt(2*V(j)^2)*V(k)^2*(V(k)^2+V(j)^2)));
                e2=erf(((V(k)^2+V(j)^2).*(ins(:,i)-Q)-(V(k)^2*U(j,i)+V(j)^2*U(k,i)))...
                    /(sqrt(2*V(j)^2)*V(k)^2*(V(k)^2+V(j)^2)));
                mul=mul.*(e1-e2);
            end
            mul=mul*temp;
            I2=I2+mul;
        end
    end
    sm = Y.*Y-2.*Y.*I1+I2;
end 