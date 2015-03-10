mean(ACC)
std(ACC)

for i = 1:numel(Timing)
    Timing_total(i) = Timing{i}.GMRAClassifier + Timing{i}.GMRAClassifierTest;
    Timing_GMRA(i) = Timing{i}.GW;
    Timing_LOL(i) = Timing{i}.LOL;
end

mean(Timing_total(:))
mean(Timing_GMRA(:))
mean(Timing_LOL(:))


