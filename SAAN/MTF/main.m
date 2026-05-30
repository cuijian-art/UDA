clc
clear 
close all
load('DataX.mat')
[a,b]=size(DataX);

for i=1:b
    data=DataX(:,i)'; 
    M=MTF(data);  
    imagesc(M)     
    colormap(jet);    

    set(gcf,'position',[0,0,300,300])
    
    
    set(gca,'XTick',[], 'YTick',[])
    axis off;  
    box off;   
    set(gca,'Position',[0 0 1 1]) 
    set(gca,'YDir','normal');
    print(gcf, ['.\jpg格式\', num2str(i), '.jpg'], '-djpeg', '-r300'); 

end