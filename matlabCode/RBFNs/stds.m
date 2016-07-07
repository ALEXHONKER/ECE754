%% STD function
function [st] = stds(var)
    if size(var,1)==1
        me=mean(var);
        su=0;
        for i=1:size(var,2)
            su=su+(var(1,i)-me)^2;
        end
        st=sqrt(su/size(var,2));
    elseif size(var,2)==1
        me=mean(var);
        su=0;
        for i=1:size(var,1)
            su=su+(var(i,1)-me)^2;
        end
        st=sqrt(su/size(var,1));        
    else
        st=[];
        me=mean(var);
        su=zeros(1,size(var,2));
        for i=1:size(var,1)
            var(i,:)=var(i,:)-me;
        end
        var=var.*var;
        su=sum(var);
        su=su./size(var,1);
        st=sqrt(su);
    end

end