function [ task, ks] = set_task_LOL( Opts, dim )

task = {};
task.LOL_alg = Opts.LOL_alg;
task.ntrain = dim;      % task.ntrain = cp.TrainSize(1);
ks=unique(floor(logspace(0,log10(task.ntrain),task.ntrain/10)));

end

