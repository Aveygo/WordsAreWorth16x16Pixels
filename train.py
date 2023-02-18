from utils import set_seed, latest_state_dict, get_model, MODEL_NAMES
from session import TrainingSession
from dataset import TokenDataset

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import numpy as np

ACCUM_ITER = 4
BATCH_SIZE = 2
LOG_VAL_LOSS_EVERY = 10

set_seed(43)

def train(model_name, epochs=1):

    model = get_model(model_name).cuda()
    
    # Uncomment to load latest state dict
    # model.load_state_dict(latest_state_dict(model_name))

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    dataset = TokenDataset()

    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-100, 100])

    dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    mem = sum([param.nelement()*param.element_size() for param in model.parameters()]) + sum([buf.nelement()*buf.element_size() for buf in model.buffers()])

    print(f"Total number of parameters: {round(sum(p.numel() for p in model.parameters())/1000000)} million.")
    print(f"Model memory usage: {round(mem/1000000000, 2)}GB")
    print(f"Training for {epochs} epochs ({epochs * len(dataloader)} samples), with a batch size of {ACCUM_ITER * BATCH_SIZE}")

    session = TrainingSession(model_name)

    for epoch in range(1):

        training_losses = []

        for i, (x, y) in enumerate(dataloader):
            x = x.cuda().long()
            y = y.cuda().long()

            with torch.set_grad_enabled(True):

                logits = model(x)      

                loss = criterion(logits, y)
                training_losses.append(loss.item())
                loss = loss / ACCUM_ITER

                loss.backward()

                if (i + 1) % ACCUM_ITER == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()    

            if (i + 1) % ACCUM_ITER == 0:
                # Log training loss
                print(f"Epoch: {epoch}, Step: {i}, Loss: {np.mean(training_losses)}")
                session.log(np.mean(training_losses), epoch * len(dataloader) + i, "train")
                session.plot("train")

                training_losses = []

            if (i + 1) % (ACCUM_ITER * LOG_VAL_LOSS_EVERY) == 0:
                # Log test loss
                with torch.no_grad():
                    loss = 0
                    for j, (x, y) in enumerate(test_dataloader):
                        x = x.cuda().long()
                        y = y.cuda().long()
                        logits = model(x)      
                        loss += criterion(logits, y).item()

                    loss /= len(test_dataloader)
                    print(f"Epoch: {epoch}, Step: {i}, Test Loss: {round(loss, 4)}")
                    session.log(loss, (epoch * len(dataloader) + i) * LOG_VAL_LOSS_EVERY, "test")
                    session.plot("test")
        
        session.save(model.state_dict())

train(MODEL_NAMES[0])