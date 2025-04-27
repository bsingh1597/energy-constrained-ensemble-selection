import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import timeit
from carbontracker.tracker import CarbonTracker
import copy
# from earlyStopping import EarlyStopping

def trainModel(device, model, model_name, image_datasets, dataloaders, criterion, optimizer, scheduler, num_epochs=10):
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Initialize carbon tracker
    tracker = CarbonTracker(
        epochs=num_epochs,
        log_dir=f"./logs/{model_name}",
        epochs_before_pred=40,
        decimal_precision=3
    )
    tracker.set_api_keys({"electricitymaps":"DSxt7q2TNdaGD"})
    # Early stopping initialization
    # early_stopping = EarlyStopping(patience=20, verbose=True)

    # Define the carbon tracker
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        tracker.epoch_start()
        startTime = timeit.default_timer()
        
        for phase in ['train', 'val']:
            print(f"****Executing phase {phase}")
            model.train(phase == 'train')
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        endTime = timeit.default_timer()
        print(f"Time: {endTime - startTime:.2f}s")
        
        scheduler.step()
        tracker.epoch_end()
    tracker.stop()
    
    model.load_state_dict(best_model_wts)
    return model
