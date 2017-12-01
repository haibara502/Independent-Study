local status, cunn = pcall(require, 'fbcunn')
if not status then
	status, cunn = pcall(require, 'cunn')
	if not status then
		print("ERROR: Could not find cunn or fbcunn.")
		os.exit()
	end
end
LookupTable = nn.LookupTable
require 'cutorch'
require 'utils/util'
require 'utils/queue'
require 'utils/heap'
require 'utils/reader'
require 'lstm'
require 'rnn'
require 'gru'

--[[
#########################################################
      Deep LSTM/RNN/GRU LM implementation via torch        
                  Wengong Jin（金汶功）                  
               Email: acmgokun@gmail.com                 
        Speech Lab, Shanghai Jiao Tong University        
#########################################################
--]]

local cmd = torch.CmdLine()

cmd:text('General Options:')
cmd:option('-train', '', 'training set file')
cmd:option('-valid', '', 'validation set file')
cmd:option('-ppl', '', 'ppl test set file')
cmd:option('-rescore', '', 'rescore file')
cmd:option('-read_model', '', 'read model from this file')
cmd:option('-print_model', '', 'print model to this file')
cmd:option('-vocab', '', 'read vocab from this file')
cmd:option('-print_vocab', '', 'print vocab to this file')
cmd:option('-trace_level', 1, 'trace level')

cmd:text('Model Options:')
cmd:option('-rnn_type', 'lstm', 'recurrent type: lstm/rnn/gru')
cmd:option('-hidden_size', 300, 'hidden layer dimension')
cmd:option('-layers', 1, 'number of recurrent layers')

cmd:text('Runtime Options:')
cmd:option('-deviceId', 1, 'train model on ith gpu')
cmd:option('-random_seed', 7, 'set initial random seed')

cmd:text('Training Options:')
cmd:option('-alpha', 0.08, 'initial learning rate')
cmd:option('-beta', 0, 'regularization constant')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-dropout', 0, 'dropout rate at each non-recurrent layer')
cmd:option('-batch_size', 32, 'number of minibatch')
cmd:option('-bptt', 10, 'back propagation through time')
cmd:option('-alpha_decay', 0.6, 'alpha *= alpha_decay if no improvement on validation set')
cmd:option('-init_weight', 0.1, 'all weights will be set to [-init_weight, init_weight] during initialization')
cmd:option('-max_norm', 50, 'threshold of gradient clipping (2-norm)')
cmd:option('-max_epoch', 20, 'max number of epoch')
cmd:option('-min_improvement', 1.01, 'start learning rate decay when improvement less than threshold')
cmd:option('-shuffle', 1, 'whether to shuffle data before each epoch')

local options = cmd:parse(arg)

if options.deviceId > 0 then
	cutorch.setDevice(options.deviceId)
else
	options.deviceId = chooseGPU()
	cutorch.setDevice(options.deviceId)
	local device_params = cutorch.getDeviceProperties(options.deviceId)
	local computability = device_params.major * 10 + device_params.minor
	if computability < 35 and options.trace_level > 0 then
		print("WARNING: fbcunn requires GPU with cuda computability >= 3.5, falling back to cunn.")
	else
		use_fbcunn = true
		LookupTable = nn.LookupTableGPU
	end
end
random_seed(options.random_seed)

local vocab = Vocab()
if options.vocab == '' then
	vocab:build_vocab(options.train)
else
	vocab:build_vocab(options.vocab)
end

options.vocab_size = vocab:vocab_size()
options.vocab = vocab
if options.print_vocab ~= '' then
	options.vocab:save(options.print_vocab)
end
if options.trace_level > 0 then
	cmd:log('/dev/null', options)
	io.stdout:flush()
end

local lm
if options.rnn_type == 'lstm' then
	lm = LSTM(options)
elseif options.rnn_type == 'gru' then
	lm = GRU(options)
else
	lm = RNN(options)
end
lm:init(options.read_model)

local start_time = torch.tic()
local alpha_decay = false
if options.train ~= '' and options.valid ~= '' then
	local len, best_ppl= lm:evaluate(options.valid)
	print('Epoch 0 validation result: words = ' .. len .. ', perplexity = ' .. string.format('%.3f', best_ppl))
	io.stdout:flush()
	lm:save_model(options.print_model)
	for iter = 1, options.max_epoch do
		print('Start training epoch ' .. iter .. ', learning rate: ' .. string.format("%.3f", options.alpha))
		if options.shuffle == 1 then
			os.execute('shuf --random-source=.randfile -o ' .. options.train .. ' ' .. options.train)
		end
		lm:train_one_epoch(options.train)
		len, valid_ppl = lm:evaluate(options.valid)
		print('Epoch ' .. iter .. ' validation result: tested words = ' .. len .. ', perplexity = ' .. string.format("%.3f", valid_ppl))
		if alpha_decay or best_ppl / valid_ppl < options.min_improvement then
			options.alpha = options.alpha * options.alpha_decay
			alpha_decay = true
		elseif iter == options.max_epoch then
			options.max_epoch = options.max_epoch + 1
		end
		if best_ppl < valid_ppl then
			lm:restore(options.print_model)
			print('Model is restored to previous epoch.')
		else 
			best_ppl = valid_ppl
			lm:save_model(options.print_model)
		end
		io.stdout:flush()
	end
	local elapsed_time = torch.toc(start_time) / 60
	print('Training finished, elapsed time = ' .. string.format('%.1f', elapsed_time))
end

if options.ppl ~= '' then
	local len, ppl = lm:evaluate(options.ppl)
	print('Test set result: tested words = ' .. len .. ', perplexity = ' .. string.format('%.3f', ppl))
end
if options.rescore ~= '' then
	lm:rescore(options.rescore)
end
