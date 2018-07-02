import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from person_pair import person_pair
from ggnn import GGNN
from torch.distributions import Bernoulli
from vgg_v1 import vgg16_rois_v1
import math

class GRM(nn.Module):
	def __init__(self, num_class = 3,
				ggnn_hidden_channel = 4098,
				ggnn_output_channel = 512, time_step = 3,
				attr_num = 80, adjacency_matrix=''):
		super(GRM, self).__init__()
		self._num_class = num_class
		self._ggnn_hidden_channel = ggnn_hidden_channel
		self._ggnn_output_channel = ggnn_output_channel
		self._time_step = time_step
		self._adjacency_matrix = adjacency_matrix
		self._attr_num = attr_num
		self._graph_num = attr_num + num_class
		

		self.fg = person_pair(num_class)

		self.full_im_net = vgg16_rois_v1(pretrained=False)

		self.ggnn = GGNN( hidden_state_channel = self._ggnn_hidden_channel,
			output_channel = self._ggnn_output_channel,
			time_step = self._time_step,
			adjacency_matrix=self._adjacency_matrix,
			num_classes = self._num_class)

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(self._ggnn_output_channel * (self._attr_num + 1) , 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1)
		)

		self.ReLU = nn.ReLU(True)

		self._initialize_weights()

	def forward(self, union, b1, b2, b_geometric, full_im, rois, categories):
		batch_size = union.size()[0]
		# full image
		rois_feature = self.full_im_net(full_im, rois, categories)
		contextual = Variable(torch.zeros(batch_size, self._graph_num, self._ggnn_hidden_channel), requires_grad=False).cuda()
		contextual[:, 0:self._num_class, 0] = 1.
		contextual[:, self._num_class:, 1] = 1.

		start_idx = 0
		end_idx = 0

		for b in range(batch_size):
			cur_rois_num = categories[b, 0].data[0]
			end_idx += cur_rois_num
			idxs = categories[b, 1:(cur_rois_num+1)].data.tolist()
			for i in range(cur_rois_num):
				contextual[b, int(idxs[i])+self._num_class, 2:] = rois_feature[start_idx+i, :]
			start_idx = end_idx

		# first glance scores
		scores, fc7_feature = self.fg(union, b1, b2, b_geometric)
		
		# ggnn input
		fc7_feature_norm_enlarge = fc7_feature.view(batch_size, 1, -1).repeat(1, self._num_class, 1)
		contextual[:, 0: self._num_class, 2:] = fc7_feature_norm_enlarge
		ggnn_input = contextual.view(batch_size, -1)

		#ggnn forward
		ggnn_feature = self.ggnn(ggnn_input)

		ggnn_feature_norm = ggnn_feature.view(batch_size * self._num_class, -1)

		#classifier
		final_scores = self.classifier(ggnn_feature_norm).view(batch_size, -1)
		
		return final_scores

	def _initialize_weights(self):

		for m in self.classifier.modules():
			cnt = 0
			if isinstance(m, nn.Linear):
				if cnt == 0:
					m.weight.data.normal_(0, 0.001)
				else :
					m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
				cnt += 1
