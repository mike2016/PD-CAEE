function plotMeanVar(data,fr,er)
    %data: len * dim
    
    dMean = mean(data,2)';
    
    dMax = max(data,[],2)';
    
    dMin = min(data,[],2)';
    x = 1:length(dMin);
    
    xx = [x,fliplr(x)];
    yy = [dMax,fliplr(dMin)];
    
    fill(xx,yy,fr,'Edgecolor',er,'FaceAlpha',0.6,'LineStyle','--')
    
    %area(dMax,'facecolor',r,'edgecolor','none','FaceAlpha',0.6);
    %hold on
    %area(dMin,'facecolor','[1,1,1]','edgecolor','none','FaceAlpha',1);
    %plot(dMean,'LineWidth',3)
   
end