
load('result_temp/weight_loss_result_dim5_832_test_LC_plus.mat')

s = 5;
E = 21;
L = E-s+1;


[M,C,I,G,W] = size(result);

figure()

wList = 1:4;
i = 1;
for j = wList
    subplot(2,2,i)
    i = i+1;
    data = reshape(result(j,[1,9,10],:,:),[3*50,21])';
    fr1 = [0.46,0.63,0.38];
    er1 = '[0.00,0.50,0.00]';

    data2 = reshape(result(j,2:8,:,:),[7*50,21])';
    fr2 = [0.88,0.49,0.67];
    er2 = '[1.00,0.00,1.00]';

    plotMeanVar(data2,fr2,er2)
    hold on
    plotMeanVar(data,fr1,er1)
    grid on
    
end
legend('Normal','HIF')


% legs = {'H','1','2','3','4','5','6','7','CS','LS'};
% figure
% for i =1:C
%     for j =1:M
%         subplot(2,4,j)
%         %d11 = M1*(reshape(d1(i,:,:),[2,50*L])-Z1);
%         %d = reshape(result(j,i,[1,4],:,s:E),[2,G*L]);
%         %scatter(d(1,:),d(2,:))
%         if i >1 && i < 9
%             sPoint = s;
%             ePoint = E;
%         else
%             sPoint = 1;
%             ePoint = 21;
%         end
%         d1 = reshape(result(j,i,3,:,s:E),[1,G*L]);
%         d2 = reshape(result(j,i,4,:,s:E),[1,G*L]);
%         scatter(d1,d2)
% %         set(gca, 'XScale', 'log')
% %         set(gca, 'YScale', 'log')
%         hold on
%         legend(legs)
%         
%     end
% end
% 
% 
% legs = {'H','1','2','3','4','5','6','7','CS','LS'};
% figure
% for i =1:C
%     for j =1:M
%     
%         if i >1 & i < 9
%             r = 'r';
%         else
%             r = 'g';
%         end
%         if i >1 && i < 9
%             sPoint = s;
%             ePoint = E;
%         else
%             sPoint = 1;
%             ePoint = 21;
%         end
%         subplot(2,4,j)
%         %d11 = M1*(reshape(d1(i,:,:),[2,50*L])-Z1);
%         %d = reshape(result(j,i,[1,4],:,s:E),[2,G*L]);
%         %scatter(d(1,:),d(2,:))
%         d1 = reshape(result(j,i,3,:,s:E),[1,G*L]);
%         d2 = reshape(result(j,i,4,:,s:E),[1,G*L]);
%         scatter(d1,d2,r)
% %         set(gca, 'XScale', 'log')
% %         set(gca, 'YScale', 'log')
%         hold on
%         legend(legs)
%         
%     end
% end
% 
% legs = {'H','1','2','3','4','5','6','7','CS','LS'};
% figure
% for i =1:C
%     for j =1:M
%     
%      
%         subplot(2,4,j)
%         d1 = reshape(result(j,i,3,1,:),[1,21]);
%         d2 = reshape(result(j,i,4,1,:),[1,21]);
%         r = sqrt(d1.^2 + d2.^2);
%         plot(r)
%         hold on
%         legend(legs)
%     end
% end
