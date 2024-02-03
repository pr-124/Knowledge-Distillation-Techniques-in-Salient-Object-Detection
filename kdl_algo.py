import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import dataset
from net  import GCPANet
from compact_net import CGCPANet
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lib.lr_finder import LRFinder
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset
from net  import GCPANet
import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lib.lr_finder import LRFinder
import numpy as np
import matplotlib.pyplot as plt

TAG = "kd"
SAVE_PATH = "/content/GCPANet/save"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")



cfg    = dataset.Config(datapath='/content/GCPANet/data/DUTS/', savepath=SAVE_PATH, mode='train', batch=8, lr=0.05, momen=0.9, decay=5e-4, epoch=30)
data   = dataset.Data(cfg)
loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
prefetcher = DataPrefetcher(loader)

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    base, head = [], []
    for name, param in teacher.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer   = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        print("in epoch")
        running_loss = 0.0
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        inputs, labels = prefetcher.next()
        while inputs is not None:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = sum(list(teacher(inputs)))

            # Forward pass with the student model
            student_logits = sum(list(student(inputs)))

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            inputs, labels = prefetcher.next()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.

nn_deep = GCPANet(cfg)
new_nn_light = CGCPANet(cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nn_deep = nn_deep.to(device)
new_nn_light = new_nn_light.to(device)

train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=prefetcher, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
# test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

# Compare the student test accuracy with and without the teacher, after distillation
# print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
# print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
# print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# def load_teacher_model(cfg):
#     teacher_model = GCPANet(cfg)
#     teacher_model.load_state_dict(torch.load('/content/GCPANet/model-30'))
#     teacher_model.eval()
#     teacher_model.cuda()
#     return teacher_model

# def initialize_student_model(cfg):
#     student_model = CGCPANet(cfg)
#     student_model.initialize()
#     student_model.train()
#     student_model.cuda()
#     return student_model


# def distillation_loss(student_output, teacher_output, temperature, alpha):
#     # Soften the outputs
#     soft_teacher_output = F.softmax(teacher_output / temperature, dim=1)
#     soft_student_output = F.log_softmax(student_output / temperature, dim=1)

#     return nn.KLDivLoss(reduction='batchmean')(soft_student_output, soft_teacher_output) * (temperature ** 2) * alpha

# def train_student_model(teacher_model, student_model, dataloader, temperature, alpha):
#     optimizer = optim.Adam(student_model.parameters(), lr=0.001)
#     criterion = nn.BCEWithLogitsLoss() 

#     for epoch in range(num_epochs):
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.cuda(), targets.cuda()

#             # Forward pass through the teacher model
#             with torch.no_grad():
#                 teacher_outputs = teacher_model(inputs)

#             # Forward pass through the student model
#             student_outputs = student_model(inputs)

#             # Calculate distillation loss
#             dist_loss = distillation_loss(student_outputs, teacher_outputs, temperature, alpha)

#             # Calculate task-specific loss
#             task_loss = criterion(student_outputs, targets)

#             # Combine losses
#             loss = dist_loss + (1 - alpha) * task_loss

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# if __name__ == "__main__":


#     teacher_model = load_teacher_model(cfg)
#     student_model = initialize_student_model(cfg)

#     # Define dataloader, num_epochs, temperature, and alpha
#     cfg    = dataset.Config(datapath='/content/GCPANet/data/DUTS/', savepath=SAVE_PATH, mode='train', batch=8, lr=0.05, momen=0.9, decay=5e-4, epoch=30)
#     data   = dataset.Data(cfg)
#     loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
#     prefetcher = DataPrefetcher(loader)
#     num_epochs = 20   # Example: set the number of epochs
#     temperature = 2.0 # Hyperparameter for distillation
#     alpha = 0.5       # Balance between distillation and task-specific loss

#     train_student_model(teacher_model, student_model, prefetcher, temperature, alpha)
