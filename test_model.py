import torch
from ClassModel import ActionClassifier
from ClassModel import input_size, hidden_size1, hidden_size2, output_size



model = ActionClassifier(input_size, hidden_size1, hidden_size2, output_size)
model.load_state_dict(torch.load('action_classifier.pth'))
model.eval()
