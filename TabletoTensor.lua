
local TabletoTensor, parent = torch.class('nn.TabletoTensor', 'nn.Module')

function TabletoTensor:__init()
   parent.__init(self)
   self.gradInput = {}
end

-- input is a table with 1-d tensors. 
function TabletoTensor:updateOutput(input)
   self.output:resize(input[1]:size()[1],input[1]:size()[2] , #input)
   -- print(input[1]:size())
  -- print(input)
  for i=1,#input do
    for j = 1, input[1]:size()[1] do
      for k = 1, input[1]:size()[2] do
        self.output[j][k][i] = input[i][j][k]
      end
    end
  end
  return self.output
end

function TabletoTensor:updateGradInput(input, gradOutput)
  self.gradInput = self.gradInput or {}
  -- print(gradOutput)
  for i=1,#input do
    self.gradInput[i] = self.gradInput[i] or input[i].new() 
    -- print(self.gradInput[i])
    self.gradInput[i]:resizeAs(input[i])
    for j = 1, input[1]:size()[1] do
      for k = 1, input[1]:size()[2] do
        self.gradInput[i][j][k] = gradOutput[j][k][i]
      end
    end 
  end
  for i=#input + 1, #self.gradInput do
    self.gradInput[i] = nil 
  end
  
  return self.gradInput
end