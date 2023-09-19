import matplotlib.pyplot as plt
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# randint_trainL1, randint_testL1 = torch.load(r'result\loss_randint_150_300_acrossSub.data1')
# randint_trainL2, randint_testL2 = torch.load(r'result\loss_randint_450_600_acrossSub.data1')
# randint_trainL = torch.cat([randint_trainL1,randint_trainL2],dim=0)
# randint_testL = torch.cat([randint_testL1,randint_testL2],dim=0)
# randint_x = range(150,150+600)
randint0_trainL1, randint0_testL1 = torch.load(r'4classloss_combRes50_randint_lr1_bs32_acrossSub_run2.data1')
randint0_trainL1acc, randint0_testL1acc = torch.load(r'4classacc_combRes50_randint_lr1_bs32_acrossSub_run2.data1')
# lr = torch.load(r'lr_combRes18_randint_lr1_bs32_acrossSub.data1')

# randint0_trainL2, randint0_testL2 = torch.load(r'result\loss_randint_int0_300_acrossSub.data1')
# randint0_trainL3, randint0_testL3 = torch.load(r'result\loss_randint_int0_600_acrossSub.data1')
# randint0_trainL4, randint0_testL4 = torch.load(r'result\loss_randint_int0_900_acrossSub.data1')
# randint0_trainL = torch.cat([randint0_trainL1,randint0_trainL2, randint0_trainL3,randint0_trainL4],dim=0)
# randint0_testL = torch.cat([randint0_testL1,randint0_testL2, randint0_testL3,randint0_testL4],dim=0)
randint0_x = range(0,100)

# plt.plot(randint_x,randint_trainL, label='train_loss_randint')
# plt.plot(randint_x,randint_testL, label='test_loss_randint')
plt.plot(randint0_x,randint0_trainL1, label='train_loss_randint0')
plt.plot(randint0_x,randint0_testL1, label='test_loss_randint0')
plt.plot(randint0_x,randint0_trainL1acc, label='train_acc_randint0')
plt.plot(randint0_x,randint0_testL1acc, label='test_acc_randint0')
# plt.plot(randint0_x,lr, label='lr')
plt.legend()
plt.show()