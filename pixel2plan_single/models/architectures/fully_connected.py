import torch.nn as nn


class FullyConnected(nn.Module):
	def __init__(self, in_features, out_features, small, device="cuda"):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.device = device

		if small:
			self.layers = nn.Sequential(
				nn.Linear(self.in_features, 128),
				nn.Tanh(),
				nn.Linear(128, 32),
				nn.Tanh(),
				nn.Linear(32, 8),
				nn.Tanh(),
				nn.Linear(8, self.out_features),
				nn.Tanh()
			).to(self.device)
		else:
			self.layers = nn.Sequential(
				nn.Linear(self.in_features, 128),
				nn.Tanh(),
				nn.Linear(128, 512),
				nn.Tanh(),
				nn.Linear(512, 256),
				nn.Tanh(),
				nn.Linear(256, 128),
				nn.Tanh(),
				nn.Linear(128, 32),
				nn.Tanh(),
				nn.Linear(32, 8),
				nn.Tanh(),
				nn.Linear(8, self.out_features),
				nn.Tanh()
			).to(self.device)

	def forward(self, x):
		"""
			needs already flatten input
		"""
		return self.layers(x)

	@property
	def state(self):
		state_dict = self.state_dict().copy()
		for k, v in state_dict.items():
			state_dict[k] = v.cpu()
		return state_dict

	@state.setter
	def state(self, state_dict):
		self.load_state_dict(state_dict)

	def init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))