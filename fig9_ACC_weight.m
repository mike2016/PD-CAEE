x = [0,1e-4,0.1,0.5,0.9,0.95,0.99,0.999,0.9999,1];
acc = [0.660,0.669,0.861,0.884,0.930, 0.929,0.933,0.929,0.841,0.507]*100;

figure()
bar(1:length(acc),acc,'FaceColor','[0.95,0.53,0.61]')
grid on
set(gca,'XTickLabel',x);
set(gca,'FontSize',20);
xlabel('$\omega$')
ylabel('Acc (%)')