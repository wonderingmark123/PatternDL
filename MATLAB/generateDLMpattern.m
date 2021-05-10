for i = 1:62
    temp =reshape( PatternFInal(1,i,:,:) , [112,112]);
    
    dlmwrite(['./PatternDL',num2str(i,'%2.2d'),'.txt'],temp,' ');
end