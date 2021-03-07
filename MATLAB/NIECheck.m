%check error
subplot 321
imagesc(PmeanPython);
colorbar('LineWidth',1.5);
set(gca,'YDir','normal');
colormap(flipud(othercolor('RdBu6')));
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);
subplot 322
imagesc(PmeanMatlab);
colorbar('LineWidth',1.5);
set(gca,'YDir','normal');
colormap(flipud(othercolor('RdBu6')));
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

subplot 323
imagesc(PIpython);
colorbar('LineWidth',1.5);
set(gca,'YDir','normal');
colormap(flipud(othercolor('RdBu6')));
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);
subplot 324
imagesc(PImeanMatlab);
colorbar('LineWidth',1.5);
set(gca,'YDir','normal');
colormap(flipud(othercolor('RdBu6')));
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

subplot 325
plot(1:length(Imatlab),Imatlab,'*','MarkerSize',5,'LineWidth',1.5);
hold on
plot(1:length(Imatlab),Ipython,'o-','MarkerSize',5,'LineWidth',1.5);
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

subplot 326
imagesc(PImeanMatlab-PmeanMatlab.*mean(Imatlab));
% imagesc(PIpython-PmeanPython.*mean(Ipython));
colorbar('LineWidth',1.5);
set(gca,'YDir','normal');
colormap(flipud(othercolor('RdBu6')));
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);