require 'luarocks.loader'
require 'nngraph'
require 'cunn'

local TRAIN_LOG_WORDS = 100000

local LSTM = torch.class('LSTM')

function LSTM:__init(options)
	self.options = options
end

-- lstm cell activation function
-- no peephole connection.
function LSTM:lstm(input, prev_c, prev_h)
	-- every call to input_hidden_sum() creates two nn.Linear() modules
	-- input, forget, output gates use different weight matrices
	-- bias is automatically included in nn.Linear()
	-- nn.Linear() is a module and nn.Linear()() is a graph node in nngraph
	local function input_hidden_sum()
		local w_i2h = nn.Linear(self.options.hidden_size, self.options.hidden_size, false)
		local w_h2h = nn.Linear(self.options.hidden_size, self.options.hidden_size) 
		return nn.CAddTable()({w_i2h(input), w_h2h(prev_h)}) --w_i2h(input) is a node!
	end
	local input_gate = nn.Sigmoid()(input_hidden_sum()) 
	local forget_gate = nn.Sigmoid()(input_hidden_sum())
	local cell_input = nn.Tanh()(input_hidden_sum())
	local cell = nn.CAddTable()({nn.CMulTable()({input_gate, cell_input}), nn.CMulTable()({forget_gate, prev_c})})
	local output_gate = nn.Sigmoid()(input_hidden_sum())
	local hidden = nn.CMulTable()({output_gate, nn.Tanh()(cell)})
	return cell, hidden --lstm returns two nodes!
end

function LSTM:build_net()
	local input = nn.Identity()() 
	local target = nn.Identity()()
	local prev_state = nn.Identity()() -- saves hidden states at all layers, hidden state includes cell activation and hidden state activation
	local next_state = {}
	local wvec = LookupTable(self.options.vocab_size, self.options.hidden_size)(input)
	local net = {[0] = wvec}
	local prev_split = {prev_state:split(2 * self.options.layers)} -- each hidden layer is split, one for cell, one for hidden, there will be two successors of nn.Identity()() if nn.Identity()() is split.
	for i = 1, self.options.layers do
		local prev_cell = prev_split[2 * i - 1]
		local prev_hidden = prev_split[2 * i]
		local dropped_input = nn.Dropout(self.options.dropout)(net[i - 1])
		local next_cell, next_hidden = self:lstm(dropped_input, prev_cell, prev_hidden)
		table.insert(next_state, next_cell)
		table.insert(next_state, next_hidden)
		net[i] = next_hidden
	end
	local dropped_hidden = nn.Dropout(self.options.dropout)(net[self.options.layers])
	local output = nn.Linear(self.options.hidden_size, self.options.vocab_size)(dropped_hidden)
	local log_prob = nn.LogSoftMax()(output)
	log_prob:annotate({["name"] = "log_prob"})
	local err = nn.ClassNLLCriterion()({log_prob, target})
	local model = nn.gModule({input, target, prev_state}, {err, nn.Identity()(next_state)}) -- input to the network (at a certain time t) is input, prev_state and target output of the network (at a certain time t) is err and next_state. (err is not a node, but next_state is changed to a node)
	model:getParameters():uniform(-self.options.init_weight, self.options.init_weight)
	return model
end

function LSTM:init(input_model)
	if input_model == '' then
		self.core_model = self:build_net()
	else
		if self.options.trace_level > 0 then
			print('Loading model from ' .. input_model)
		end
		self:load_model(input_model)
	end
	self.core_model = transfer2gpu(self.core_model)
	self.params, self.grads = self.core_model:getParameters()
	self.models = make_recurrent(self.core_model, self.options.bptt)

	self.history = {}
	self.grad_h = {}
	self.tmp_hist = {}
	self.err = transfer2gpu(torch.zeros(self.options.bptt))
	for i = 0, self.options.bptt do
		self.history[i] = {}
		self.tmp_hist[i] = {}
		for j = 1, 2 * self.options.layers do
			self.history[i][j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size)) -- torch tensor is row major, thus batch is set to row
			self.tmp_hist[i][j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size)) -- tmp_hist serves as backup of self.history, it is important for correctness
		end
	end
	for i = 1, 2 * self.options.layers do
		self.grad_h[i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size))
	end
end

function LSTM:forward(output_layer)
	local len = 0
	local log_prob = 0
	replace(self.history[0], self.history[self.options.bptt])
	for i = 1, self.options.bptt do
		local input = self.cur_batch[i]
		local target = self.cur_batch[i + 1]
		replace(self.tmp_hist[i - 1], self.history[i - 1]) -- self.history[i - 1] cannot be modified, since it is used for output->hidden as well as hidden->hidden, thus copy it to tmp_hist
		for j = 1, self.options.batch_size do
			if self.options.vocab:is_eos(input[j]) then
				for k = 1, self.options.layers * 2 do
					self.tmp_hist[i - 1][k][j]:zero() -- clear the hidden state at the end of a sentence
				end
			end
		end
		self.err[i], self.history[i] = unpack(self.models[i]:forward({input, target, self.tmp_hist[i - 1]}))
		local prob = output_layer[i].output
		for j = 1, self.options.batch_size do
			if not self.options.vocab:is_null(target[j]) then
				log_prob = log_prob - prob[j][target[j]]
				len = len + 1
			end
		end
	end
	return len, log_prob
end

function LSTM:backward()
	self.grads:mul(self.options.momentum / (-self.options.alpha)) -- normal momentum, alpha will be multiplied back
	reset(self.grad_h) -- clear the gradient at the end of the sequence.
	for i = self.options.bptt, 1, -1 do
		local input = self.cur_batch[i]
		local target = self.cur_batch[i + 1]
		local derr = transfer2gpu(torch.ones(1):mul(self.options.batch_size))
		local grad_h = self.models[i]:backward({input, target, self.tmp_hist[i - 1]}, {derr, self.grad_h})[3] 
		replace(self.grad_h, grad_h) --grad_h refers to some place in self.grads, thus need a copy as well.
		for j = 1, self.options.batch_size do
			if self.options.vocab:is_eos(input[j]) then
				for k = 1, self.options.layers * 2 do
					self.grad_h[k][j]:zero() -- clear the gradient at the end of a sentence
				end
			end
		end
	end
	local grad_norm = self.grads:norm()
	if grad_norm > self.options.max_norm then 
		self.grads:mul(self.options.max_norm / grad_norm) --clip gradient in case
	end
	if self.options.beta > 0 then
		self.params:mul(1 - self.options.beta) -- regularization
	end
	self.params:add(self.grads:mul(-self.options.alpha))
end

function LSTM:train_one_epoch(train)
	self.reader = DataReader(train, self.options.batch_size, self.options.vocab)
	reset(self.history[self.options.bptt]) -- copied to self.history[0] in forward()
	self.grads:zero()
	local probs = {}
	for i = 1, self.options.bptt do
		probs[i] = find_module(self.models[i], "log_prob")
	end
	local len = 0
	local ppl = 0
	local begin_time = torch.tic()
	while true do
		self.cur_batch = self.reader:get_batch(self.options.bptt) -- the last word in each batch will become the first word in next batch
		if self.cur_batch == nil then
			break
		end
		self.cur_batch = transfer2gpu(self.cur_batch)
		local per_len, per_ppl = self:forward(probs)
		len = len + per_len
		ppl = ppl + per_ppl
		self:backward()
		if len > TRAIN_LOG_WORDS then
			local elapsed_time = torch.toc(begin_time) / 60
			if self.options.trace_level > 0 then
				print('trained words = ' .. len .. ', perplexity = ' .. string.format('%.3f', torch.exp(ppl / len)) .. ', elapsed time = ' .. string.format('%.1f', elapsed_time) .. ' mins.')
			end
			len = 0
			ppl = 0
			io.stdout:flush()
			collectgarbage()
		end
	end
end

function LSTM:evaluate(data)
	local perplexity = 0
	local len = 0

	local probs = {}
	for i = 1, self.options.bptt do
		probs[i] = find_module(self.models[i], "log_prob")
	end
	self.reader = DataReader(data, self.options.batch_size, self.options.vocab)
	reset(self.history[self.options.bptt]) -- copied to self.history[0] in forward()
	disable_dropout(self.models)
	while true do
		self.cur_batch = self.reader:get_batch(self.options.bptt)
		if self.cur_batch == nil then
			break
		end
		self.cur_batch = transfer2gpu(self.cur_batch)
		local per_len, ppl = self:forward(probs)
		perplexity  = perplexity + ppl
		len = len + per_len
	end
	enable_dropout(self.models)
	return len, torch.exp(perplexity / len)
end

function LSTM:forward2(output_layer)
	local log_prob = {}
	replace(self.history[0], self.history[1])
	local input = self.cur_batch[1]
	local target = self.cur_batch[2]
	replace(self.tmp_hist[0], self.history[0]) -- self.history[i - 1] cannot be modified, since it is used for output->hidden as well as hidden->hidden, thus copy it to tmp_hist
	for j = 1, self.options.batch_size do
		if self.options.vocab:is_eos(input[j]) then
			for k = 1, self.options.layers * 2 do
				self.tmp_hist[0][k][j]:zero() -- clear the hidden state at the end of a sentence
			end
		end
	end
	self.err[1], self.history[1] = unpack(self.models[1]:forward({input, target, self.tmp_hist[0]}))
	local prob = output_layer[1].output
	for j = 1, self.options.batch_size do
		if not self.options.vocab:is_null(target[j]) then
			log_prob[j] = prob[j][target[j]]
		end
	end
	return log_prob
end

function LSTM:rescore(data)
	local probs = {}
	local uttr_front = 1
	local uttr_tail = 1
	local uttr_prob = {}
	local uttr_list = {}
	for i = 1, self.options.bptt do
		probs[i] = find_module(self.models[i], "log_prob")
	end
	self.reader = DataReader(data, self.options.batch_size, self.options.vocab)
	reset(self.history[1])
	disable_dropout(self.models)
	while true do
		self.cur_batch = self.reader:get_batch(1)
		if self.cur_batch == nil then
			break
		end
		for i = 1, self.options.batch_size do
			if self.options.vocab:is_eos(self.cur_batch[1][i]) then
				if uttr_list[i] then
					uttr_prob[uttr_list[i]] = -uttr_prob[uttr_list[i]] --mark it as positive
				end
				uttr_list[i] = nil
				if not self.options.vocab:is_null(self.cur_batch[2][i]) then
					uttr_list[i] = uttr_tail
					uttr_tail = uttr_tail + 1
				end
			end
		end
		self.cur_batch = transfer2gpu(self.cur_batch)
		local prob_table = self:forward2(probs)
		for i = 1, self.options.batch_size do
			if prob_table[i] then
				uttr_prob[uttr_list[i]] = (uttr_prob[uttr_list[i]] or 0) + prob_table[i]
			end
		end
		while uttr_front < uttr_tail and uttr_prob[uttr_front] > 0 do
			io.write(string.format("%.4f\n", uttr_prob[uttr_front]))
			uttr_prob[uttr_front] = nil -- set it to nil, for garbage collection
			uttr_front = uttr_front + 1
		end
	end
	while uttr_front < uttr_tail and uttr_prob[uttr_front] do
		io.write(string.format("%.4f\n", math.abs(uttr_prob[uttr_front])))
		uttr_front = uttr_front + 1
	end
	enable_dropout(self.models)
end

function LSTM:load_model(input_file)
	self.core_model = torch.load(input_file)
end

function LSTM:restore(model)
	self:load_model(model)
	self.core_model = transfer2gpu(self.core_model)
	self.params, self.grads = self.core_model:getParameters()
	self.models = make_recurrent(self.core_model, self.options.bptt)
	collectgarbage()
end

function LSTM:save_model(output_file)
	torch.save(output_file, self.core_model)
end
