import torch
import time
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex
from resnet_archi import CPnet

class KD_loss(torch.nn.Module):
    def __init__(self, alpha, beta):
        super(KD_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_32_pred, y_32_true, y_3_pred, y_3_true):
        y_32_loss = torch.mean(y_32_true - y_32_pred)**2

        flow_loss = F.mse_loss(y_3_pred[:,:2], y_3_true[:,:2])
        flow_loss /= 2

        y_3_true_map = F.sigmoid(y_3_true[:,2])
        map_loss = F.binary_cross_entropy_with_logits(y_3_pred[:,2] ,y_3_true_map)

        y_3_loss = flow_loss + map_loss
        
        return y_32_loss * self.alpha, y_3_loss * self.beta

def trainEpoch(unet, train_loader, validation_loader, loss_fn, optimiser, scheduler, epoch_num, device, progress=True):
    time_start = time.time()
    unet.train()
    train_y_32_loss, train_map_loss, train_IoU = 0, 0, 0

    for image, upsample, cp_output in train_loader:

        if device is not None:
            (image, upsample, cp_output) = (image.to(device),upsample.to(device),cp_output.to(device)) # sending the data to the device (cpu or GPU)

        y_32_pred, map_pred = unet(image)
        y_32_pred = y_32_pred.squeeze(1)
        map_pred = map_pred.squeeze(1)

        loss_32, loss_map = loss_fn(y_32_pred,  upsample, map_pred, cp_output) # calculate the loss of that prediction
        loss = loss_32 + loss_map
        train_y_32_loss += loss_32.item()
        train_map_loss += loss_map.item()
        
        loss.backward()

        optimiser.step() # update model parameters
        optimiser.zero_grad()

        #IoU score
        jaccard = BinaryJaccardIndex(threshold=0.5).to(device)
        map_pred = F.sigmoid(map_pred)
        cp_output = F.sigmoid(cp_output)
        cp_output = torch.where(cp_output > 0.5, 1.0, 0.0)
        iou = jaccard(map_pred, cp_output)

        if not torch.isnan(iou):
            train_IoU += iou.item()
        else:
            train_IoU += 0

        del image
        del upsample
        del cp_output
        torch.cuda.empty_cache()

    if scheduler is not None:
        scheduler.step()

    train_y_32_loss, train_map_loss, train_IoU = train_y_32_loss/len(train_loader), train_map_loss/len(train_loader), train_IoU/len(train_loader)


    val_y_32_loss, val_map_loss, val_IoU = 0, 0, 0
    for image, upsample, cp_output in validation_loader:
    
        if device is not None:
            (image, upsample, cp_output) = (image.to(device),upsample.to(device),cp_output.to(device)) # sending the data to the device (cpu or GPU)

        y_32_pred, map_pred = unet(image)

        y_32_pred = y_32_pred.squeeze(1)
        map_pred = map_pred.squeeze(1)

        loss_32, loss_map = loss_fn(y_32_pred,  upsample, map_pred, cp_output) # calculate the loss of that prediction
        val_y_32_loss += loss_32.item()
        val_map_loss += loss_map.item()

        #IoU score
        jaccard = BinaryJaccardIndex(threshold=0.5).to(device)
        map_pred = F.sigmoid(map_pred)
        cp_output = F.sigmoid(cp_output)
        cp_output = torch.where(cp_output > 0.5, 1.0, 0.0)
        iou = jaccard(map_pred, cp_output)
        if not torch.isnan(iou):
            val_IoU += iou.item()
        else:
            val_IoU += 0

        del image
        del upsample
        del cp_output
        torch.cuda.empty_cache()

    val_y_32_loss, val_map_loss, val_IoU = val_y_32_loss/len(validation_loader), val_map_loss/len(validation_loader), val_IoU/len(validation_loader)
    
    #we might add displaying later on
    if progress:
        if epoch_num is None:
            print('Train 32 loss: ', train_y_32_loss,'Train map loss', train_map_loss, 'Train IoU', train_IoU, 'Val 32 loss: ', val_y_32_loss, 'Val map loss: ', val_map_loss, 'Val IoU: ', val_IoU, 'Time: ', time.time()-time_start)
        else:
            print('Epoch: ', epoch_num, 'Train 32 loss: ', train_y_32_loss,'Train map loss', train_map_loss, 'Train IoU', train_IoU, 'Val 32 loss: ', val_y_32_loss, 'Val map loss: ', val_map_loss, 'Val IoU: ', val_IoU, 'Time: ', time.time()-time_start)
        
    torch.cuda.empty_cache()

    return unet, train_y_32_loss, train_map_loss, train_IoU, val_y_32_loss, val_map_loss, val_IoU

def train_model(n_base,num_epochs,name_of_model, train_loader, validation_loader, device=None,progress=True,seed=None):    
    
    torch.manual_seed(seed)
    student_model = CPnet(nbase=n_base, nout=3, sz=3,
                residual_on=True, style_on=True, 
                concatenation=False, mkldnn=False)
    
    if device is not None:
        student_model = student_model.to(device)

    loss_fn = KD_loss(alpha=2, beta=1)
    optimiser = torch.optim.Adam(student_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)

    best_val_map_loss = 1000
    for epoch in range(num_epochs):
        student_model, train_y_32_loss, train_map_loss, train_IoU, val_y_32_loss, val_map_loss, val_IoU = trainEpoch(student_model, train_loader, validation_loader, loss_fn, optimiser, scheduler=scheduler, epoch_num=epoch, device=device, progress=progress)
        if val_map_loss < best_val_map_loss:
            best_val_map_loss = val_map_loss
            torch.save(student_model.state_dict(), name_of_model)

    return student_model


    

from data_utils import get_training_and_validation_data

if __name__ == '__main__':

    cellpose_model_directory = "/Users/rehanzuberi/Documents/Development/distillCellSegTrack/pipeline/CellPose_models/Nuclei_Hoechst"
    image_folder = "/Users/rehanzuberi/Downloads/development/distillCellSegTrack/pipeline/saved_cell_images_1237"
    
    train_loader, validation_loader = get_training_and_validation_data(cellpose_model_directory, image_folder, channel = 0)

    student_model = train_model([1,32],100,'resnet_nuc_32',train_loader, validation_loader, device='mps',progress=True,seed=23944)

    