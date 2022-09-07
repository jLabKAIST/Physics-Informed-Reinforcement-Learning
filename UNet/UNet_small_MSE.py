
import torch
import torch.nn as nn
import torch.nn.functional as F

class convrelu(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.convrelu = nn.Sequential(nn.Conv1d(nin, nout, 3, padding='same',padding_mode = 'circular'), 
                                      nn.BatchNorm1d(nout), 
                                      nn.ReLU(inplace=True)
                                      )
    def forward(self, x):
        return self.convrelu(x)
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv1d(1, 16, 3, padding='same',bias=True,padding_mode = 'circular')
        self.conv1_2 = convrelu(16, 16)
        self.conv1_3 = convrelu(16, 16)
        self.conv1_4 = convrelu(16, 16)
        self.conv1_5 = convrelu(16, 16)
        self.conv1_6 = convrelu(16, 16)
        self.pool_1 = nn.MaxPool1d(2) #non-Uniform

        self.conv2_1 = nn.Conv1d(16, 32, 3, padding='same',bias=True,padding_mode = 'circular')
        self.conv2_2 = convrelu(32, 32)
        self.conv2_3 = convrelu(32, 32)
        self.conv2_4 = convrelu(32, 32)
        self.conv2_5 = convrelu(32, 32)
        self.conv2_6 = convrelu(32, 32)
        self.pool_2 = nn.MaxPool1d(2) #non-Uniform

        self.conv3_1 = nn.Conv1d(32, 64, 3, padding='same',bias=True,padding_mode = 'circular')
        self.conv3_2 = convrelu(64, 64)
        self.conv3_3 = convrelu(64, 64)
        self.conv3_4 = convrelu(64, 64)
        self.conv3_5 = convrelu(64, 64)
        self.conv3_6 = convrelu(64, 64)
        self.pool_3 = nn.MaxPool1d(2) #Uniform (X

        self.conv4_1 = nn.Conv1d(64, 128, 3, padding='same',bias=True,padding_mode = 'circular')
        self.conv4_2 = convrelu(128, 128)
        self.conv4_3 = convrelu(128, 128)
        self.conv4_4 = convrelu(128, 128)
        self.conv4_5 = convrelu(128, 128)
        self.conv4_6 = convrelu(128, 128)
        self.pool_4 = nn.MaxPool1d(2) #Uniform (X

        self.conv6_1 = nn.Conv1d(128, 256,3, padding='same',bias=True,padding_mode = 'circular')
        self.conv6_2 = convrelu(256, 256)
        self.conv6_3 = convrelu(256, 256)
        self.conv6_4 = convrelu(256, 256)
        self.conv6_5 = convrelu(256, 256)
        self.conv6_6 = convrelu(256, 256)
        self.upsam_6 = nn.Upsample(scale_factor=2) #Uniform (X

        self.conv8_1 = nn.Conv1d(128+256, 128, 3, padding='same',bias=True,padding_mode = 'circular')
        self.conv8_2 = convrelu(128, 128)
        self.conv8_3 = convrelu(128, 128)
        self.conv8_4 = convrelu(128, 128)
        self.conv8_5 = convrelu(128, 128)
        self.conv8_6 = convrelu(128, 128)
        self.upsam_8 = nn.Upsample(scale_factor=2) #Uniform (X

        self.conv9_1 = nn.Conv1d(64+128, 64, 3, padding='same',bias=True,padding_mode = 'circular')
        self.conv9_2 = convrelu(64, 64)
        self.conv9_3 = convrelu(64, 64)
        self.conv9_4 = convrelu(64, 64)
        self.conv9_5 = convrelu(64, 64)
        self.conv9_6 = convrelu(64, 64)
        self.upsam_9 = nn.Upsample(scale_factor=2) #Uniform (X

        self.conv10_1 = nn.Conv1d(32+64, 32, 3, padding='same',bias=True,padding_mode = 'circular')
        self.conv10_2 = convrelu(32, 32)
        self.conv10_3 = convrelu(32, 32)
        self.conv10_4 = convrelu(32, 32)
        self.conv10_5 = convrelu(32, 32)
        self.conv10_6 = convrelu(32, 32)
        self.upsam_10 = nn.Upsample(scale_factor=2) #non-Uniform

        self.conv11_1 = nn.Conv1d(16+32, 16, 3, padding='same',bias=True,padding_mode = 'circular')
        self.conv11_2 = convrelu(16, 16)
        self.conv11_3 = convrelu(16, 16)
        self.conv11_4 = convrelu(16, 16)
        self.conv11_5 = convrelu(16, 16)
        self.conv11_6 = convrelu(16, 16)

        self.conv11_fin = nn.Conv1d(16, 1, 3, padding='same',bias=True,padding_mode = 'circular')
    
    def forward(self,img):
        res1_1 = self.conv1_1(img)
        temp = self.conv1_2(res1_1)
        temp = self.conv1_3(temp) + res1_1 ## sum
        res1_2 = self.conv1_4(temp)
        temp = self.conv1_5(res1_2)
        shortcut1 = self.conv1_6(temp) + res1_2
    
        temp = self.pool_1(shortcut1)

        res2_1 = self.conv2_1(temp)
        temp = self.conv2_2(res2_1)
        temp = self.conv2_3(temp) + res2_1 
        res2_2 = self.conv2_4(temp)
        temp = self.conv2_5(res2_2)
        shortcut2 = self.conv2_6(temp) + res2_2

        temp = self.pool_2(shortcut2)

        res3_1 = self.conv3_1(temp)
        temp = self.conv3_2(res3_1)
        temp = self.conv3_3(temp) + res3_1 
        res3_2 = self.conv3_4(temp)
        temp = self.conv3_5(res3_2)
        shortcut3 = self.conv3_6(temp) + res3_2

        temp = self.pool_3(shortcut3)

        res4_1 = self.conv4_1(temp)
        temp = self.conv4_2(res4_1)
        temp = self.conv4_3(temp) + res4_1 
        res4_2 = self.conv4_4(temp)
        temp = self.conv4_5(res4_2)
        shortcut4 = self.conv4_6(temp) + res4_2

        temp = self.pool_4(shortcut4)


        res6_1 = self.conv6_1(temp)
        temp = self.conv6_2(res6_1)
        temp = self.conv6_3(temp) + res6_1 
        res6_2 = self.conv6_4(temp)
        temp = self.conv6_5(res6_2)
        temp = self.conv6_6(temp) + res6_2

        temp = self.upsam_6(temp)

        temp = torch.cat([temp,shortcut4],dim = 1) ######

        res8_1 = self.conv8_1(temp)
        temp = self.conv8_2(res8_1)
        temp = self.conv8_3(temp) + res8_1 
        res8_2 = self.conv8_4(temp)
        temp = self.conv8_5(res8_2)
        temp = self.conv8_6(temp) + res8_2

        temp = self.upsam_8(temp)
        temp = torch.cat([temp,shortcut3],dim = 1) ######

        res9_1 = self.conv9_1(temp)
        temp = self.conv9_2(res9_1)
        temp = self.conv9_3(temp) + res9_1 
        res9_2 = self.conv9_4(temp)
        temp = self.conv9_5(res9_2)
        temp = self.conv9_6(temp) + res9_2

        temp = self.upsam_9(temp)
        temp = torch.cat([temp,shortcut2],dim = 1) ######

        res10_1 = self.conv10_1(temp)
        temp = self.conv10_2(res10_1)
        temp = self.conv10_3(temp) + res10_1 
        res10_2 = self.conv10_4(temp)
        temp = self.conv10_5(res10_2)
        temp = self.conv10_6(temp) + res10_2

        temp = self.upsam_10(temp)
        temp = torch.cat([temp,shortcut1],dim = 1) ######

        res11_1 = self.conv11_1(temp)
        temp = self.conv11_2(res11_1)
        temp = self.conv11_3(temp) + res11_1 
        res11_2 = self.conv11_4(temp)
        temp = self.conv11_5(res11_2)
        temp = self.conv11_6(temp) + res11_2

        temp = self.conv11_fin(temp)
        return temp


print('UNet defined')

def train(model, train_loader, optimizer):
    model.train() 
    dataloss_batch =[]
    criterion = nn.MSELoss()
    for idx, (img, grad) in enumerate(train_loader):
        img, grad = img.to(device), grad.to(device)
        optimizer.zero_grad()             
        predictions = model(img)    #####img.permute(0, 1, 2): batch-height-width-channel
        dataloss = criterion(predictions, grad)
        dataloss_batch.append(dataloss.item())
        dataloss.backward()              
        optimizer.step()
    A = np.array(dataloss_batch)
    #running_loss = sum(dataloss_batch)/len(dataloss_batch)
    running_loss = np.sqrt(np.mean(A))
    # print('train loss epoch: {:.8f}'.format(math.sqrt(running_loss)))
    return running_loss      

def test(model, test_loader):
    model.eval()
    testloss_batch =[]
    criterion = nn.MSELoss()
    with torch.no_grad():
         for idx, (img, grad) in enumerate(test_loader):
            img, grad = img.to(device), grad.to(device)
            outputs = model(img)
            testloss_batch.append(criterion(outputs, grad).item())
    B = np.array(testloss_batch)
    #test_loss = sum(testloss_batch)/len(testloss_batch)
    test_loss = np.sqrt(np.mean(B))
    return test_loss