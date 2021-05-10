% load('./PatternOrigin.mat');
%%
x=112; y=112;
% txt(:,:)=PatternPython(62,:,:);
% txt(:,:)=PatternOri(1,1,:,:);
TXT=reshape(rescale(PatternOri,0,255),[x,y]);
imshow(TXT,[]);
colorbar('LineWidth',1.5);
%% correlation
for x=50:52
    for y=50:53
        PP=zeros(1,112,112);
        P=zeros(1,112,112);
        for i=1:62
            PP=PP+PatternPython(i,:,:).*PatternPython(i,x,y);
            P=P+PatternPython(i,:,:);
        end
        PP_mean=PP/62;
        P_mean=P/62;
        correlation=PP_mean-P_mean.*mean(P_mean(:,x,y));
        correlation1(:,:)=correlation(1,:,:);
        subplot 121
        imagesc(1:112,1:112,correlation1);
        colorbar('LineWidth',1.5);
        set(gca,'YDir','normal');
        set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);
        subplot 122
        p=zeros(112,112);
        p(x,y)=1;
        imagesc(1:112,1:112,p);
        colorbar('LineWidth',1.5);
        set(gca,'YDir','normal');
        set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

    end
end