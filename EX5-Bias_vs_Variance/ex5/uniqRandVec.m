function indx = uniqRandVec(ii,m)
% function to generate a random vector of dimension ii with unique values ranging
% from 1 to m.

    indx = zeros(ii,1);
    temp = 0;
    aa = 1;
    while aa < ii+1
        temp = ceil(rand()*m);
        if (sum(temp == indx) == 0) %if value in temp is not in indx
            indx(aa) = temp;
            aa = aa + 1;
        end            
end
