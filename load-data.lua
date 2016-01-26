require 'dp'
require 'cunn'
require 'torchx'
require 'lfs'

function exists(file)
	local f=io.open(file,"r")
	if f~=nil then io.close(f) return true else return false end
end

function ReadData(dataPath)
	-- 1. load sentences into input and target Tensors
	-- local train =  -- 1
	-- local valid = paths.indexdir(paths.concat(dataPath, 'validation_m')) -- 2
	-- local test = paths.indexdir(paths.concat(dataPath, 'test_m')) -- 2
	-- local size = train:size() + test:size() + valid:size()
	local ds = {}
	local cf = dataPath .. "cachedData"
	if exists(cf) then
		ds = torch.load(cf)
		return ds
	end
	local type_names = { "train", "valid", "test"}
  print("Inside ReadData...")
	for i, name_t in ipairs(type_names) do 
		local train_dir = dataPath .. name_t 
		local index = 1
		local train_d = {}
    collectgarbage()
		print(string.format("Processing %s",train_dir ))
		local files = {}
		for file in lfs.dir(train_dir) do
			table.insert(files, file)
		end
		for i, file in ipairs(files) do
		  if file ~= "." and file ~= ".." then
			  file_n = train_dir .. '/' .. file
			  io.input(file_n)
			  -- print(file_n)
			  local datapoint = {}
			  local names = {"doc", "query", "target"}
			  for i, name in ipairs(names) do
				  local dim =  io.read("*number")
				  -- print(dim)
				  -- local tab = {}
				  local doc = torch.LongTensor(dim, 1):cuda()
				  for j = 1, dim do
				  	local x = io.read("*number")
				  	-- print(x)
				  	-- local doc = torch.LongTensor(1):cuda()
				  	doc[j][1] = x
				  	-- tab[j] = doc
				  end
				  datapoint[name] = doc
			  end
			  collectgarbage()
			  train_d[index] = datapoint
			  index = index + 1
        if index % 10000 == 0 then
          print(index)
          print(collectgarbage("count"))
        end
			  io.input():close()
        
			end
		end
		ds[name_t] = train_d
	end
	print("Saving serialized data...")
	torch.save(cf, ds)
	print("Saving of data-set complete...")
	return ds
end

-- local d = ReadData("/home/sahil/Desktop/cs/neuralproceedings/rnn/examples/")
-- print(d)
