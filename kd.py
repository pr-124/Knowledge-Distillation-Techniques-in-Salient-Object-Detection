import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

from data import dataset
from net  import GCPANet
from compact_net import CGCPANet
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lib.lr_finder import LRFinder
import torch.nn.init as init

import torch.optim as optim

TAG = "kd"
SAVE_PATH = "/content/GCPANet/save"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")

cfg    = dataset.Config(datapath='/content/GCPANet/data/DUTS/', savepath=SAVE_PATH, mode='train', batch=8, lr=0.05, momen=0.9, decay=5e-4, epoch=30)
data   = dataset.Data(cfg)
loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
prefetcher = DataPrefetcher(loader)

temperatures = 10
# trade-off between soft-target (st) cross-entropy and true-target (tt) cross-entropy;
# loss = alpha * st + (1 - alpha) * tt
alphas = 0.5
learning_rates = 1e-2
learning_rate_decays = 0.95
weight_decays = 1e-5
momentums = 0.9
dropout_probabilities = (0.0, 0.0)

def studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha):
	"""
	One training step of student network: forward prop + backprop + update parameters
	Return: (loss, accuracy) of current batch
	"""
	optimizer.zero_grad()
	teacher_pred = None
	if (alpha > 0):
		with torch.no_grad():
			teacher_pred = teacher_net(X)
	student_pred = student_net(X)
	student_pred = student_pred[0]*1 + student_pred[1]*0.8 + student_pred[2]*0.6 + student_pred[3]*0.4
	teacher_pred = teacher_pred[0]*1 + teacher_pred[1]*0.8 + teacher_pred[2]*0.6 + teacher_pred[3]*0.4
	loss = studentLossFn(teacher_pred, student_pred, y, T, alpha)
	loss.backward()
	optimizer.step()
	accuracy = float(torch.sum(torch.argmax(student_pred, dim=1) == y).item()) / y.shape[0]
	return loss, accuracy

def trainStudentOnHparam(teacher_net, student_net, num_epochs, 
						train_loader,
						print_every=0, 
						fast_device=torch.device('cpu')):
	"""
	Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch
	Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
	"""
	train_loss_list, train_acc_list, val_acc_list = [], [], []
	T = temperatures
	alpha = alphas
	# student_net.dropout_input = hparam['dropout_input']
	# student_net.dropout_hidden = hparam['dropout_hidden']
	optimizer = optim.SGD(student_net.parameters(), lr=learning_rates, momentum=momentums, weight_decay=weight_decays)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=learning_rate_decays)

	def studentLossFn(teacher_pred, student_pred, y, T, alpha):
		"""
		Loss function for student network: Loss = alpha * (distillation loss with soft-target) + (1 - alpha) * (cross-entropy loss with true label)
		Return: loss
		"""
		if (alpha > 0):
			loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha + F.binary_cross_entropy_with_logits(student_pred, y) * (1 - alpha)
			# F.binary_cross_entropy_with_logits(out2, mask)
		else:
			loss = F.cross_entropy(student_pred, y)
		return loss

	for epoch in range(num_epochs):
		print("in epoch")
		lr_scheduler.step()
		i = 0

		prefetcher = DataPrefetcher(loader)
		image, mask = prefetcher.next()
		while image is not None:
			i+=1
			X, y = image, mask
			X, y = X.to(fast_device), y.to(fast_device)
			loss, acc = studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha)
			train_loss_list.append(loss)
			train_acc_list.append(acc)
		
			if print_every > 0 and i % print_every == print_every - 1:
				print('[%d, %5d/%5d] train loss: %.3f train accuracy: %.3f' %
					  (epoch + 1, i, len(loader), loss, acc))
				
			if i % 10 == 0:
				msg = '%s | epoch=%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), epoch+1, optimizer.param_groups[0]['lr'], loss.item())
				logger.info(msg)

			image, mask = prefetcher.next()

		torch.save(student_net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))

	return {'train_loss': train_loss_list, 
			'train_acc': train_acc_list, 
			'val_acc': val_acc_list}

nn_deep = GCPANet(cfg)
nn_deep.load_state_dict(torch.load('/content/drive/MyDrive/DeepLearningAssgn/model-30'), strict=False)
new_nn_light = CGCPANet(cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nn_deep = nn_deep.to(device)
new_nn_light = new_nn_light.to(device)

trainStudentOnHparam(teacher_net=nn_deep, student_net=new_nn_light, num_epochs=10, train_loader=prefetcher, print_every = 100, fast_device=device)
