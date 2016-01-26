require 'rnn'
require 'load-data'
require 'CAddTensorTable'
require 'TabletoTensor'
require 'optim'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th rnn-word.lua > results.txt')
cmd:text('Options:')
cmd:option('--batchSize', 1, 'Batch size')
cmd:option('--epochs', 500, 'Number of times we go over data set')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--dataDir', "data/",
 		'The data\'s home')
cmd:option('--lr', 0.001, 'learning rate')
cmd:option('--minlr', 0.00001, 'minimum learning rate')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--rho', 2000, 'how many steps to back propagate through')
cmd:option('--hiddenSize', 2, 'size of hidden layer')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--vocabSize', 101686, 'vocab size')

cmd:text()
local opt = cmd:parse(arg or {})
print(opt)

function IsCorrect(outputs, target)
	aaaa, indice = torch.max(outputs, 2)
	if indice[1][1] == target then
		return 1
	else
		return 0
	end
end

function Epoch(data, mode, net, criterion)
	local correct = 0
	local errorr = 0.0
	local start = os.clock()
	if mode == 'train' then
		net:training()
	else
		net:evaluate()
	end
	net:zeroGradParameters()
	spliter = nn.SplitTable(1):cuda()
	for batch = 1, #data[mode] do
		print(data[mode][batch])
		local query = spliter:forward(data[mode][batch]['query'])
		local doc = spliter:forward(data[mode][batch]['doc'])
		local outputs = net:forward {query, doc}
		correct = correct + IsCorrect(outputs, data[mode][batch]['target'][1])
		local err = criterion:forward(outputs, data[mode][batch]['target'][1])
		errorr = errorr + err
		if mode == 'train' then
			local gradOutputs = criterion:backward(outputs, data[mode][batch]['target'][1])
			local gradInputs = net:backward({query,doc} ,gradOutputs)
	   	net:updateGradParameters(opt.momentum)
	   	net:updateParameters(opt.lr)
	   	net:maxParamNorm(opt.maxOutNorm)
		end
		net:zeroGradParameters()
		net:forget()
	end
	print(string.format("Mode: %s Accuracy: %f Error: %f Speed: %.3f examples/s", 
							mode, correct/#data[mode], errorr,#data[mode]/(os.clock() - start) ))
end

---------------------------------------------MODEL-----------------------------------------
-- query bi-LSTM
print("Building model...")
local r_q = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
local q_rnn = nn.Sequential()
   :add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
   :add(r_q)

q_rnn = nn.BiSequencer(q_rnn)
q_net = nn.Sequential()
q_net
  :add(q_rnn)
  :add(nn.SelectTable(-1))

  -- document bi-lstm
local r_d = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
local d_rnn = nn.Sequential()
   :add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
   :add(r_d)
d_rnn = nn.BiSequencer(d_rnn)
d_net = nn.Sequential()
d_net
  :add(d_rnn)

-- Combining document and query
split_input = nn.ParallelTable()
q_side = nn.Sequential()
q_side
	:add(q_net)
d_side = nn.Sequential()
d_side
	:add(d_net)
split_input
	:add(q_side)
	:add(d_side)

net = nn.Sequential()
net
	:add(split_input)
  :add(nn.SelectTable(1)) -- NOT poC
	:add(nn.Linear(2*opt.hiddenSize, opt.vocabSize)) -- SHOULD be opt.hiddenSize
	:add(nn.LogSoftMax())
net:cuda()
print(net)

-- build criterion
criterion = nn.ClassNLLCriterion():cuda()
print("Reading data...")
local d = ReadData(opt.dataDir)
print("Data read complete...")
local change = (opt.minlr - opt.lr)/opt.epochs
print("Optimizing using SGD...")
for iteration = 1, opt.epochs do
  Epoch(d, 'train', net, criterion)
  Epoch(d, 'valid', net, criterion)
  Epoch(d, 'test', net, criterion)
  opt.lr = opt.lr + change
end
