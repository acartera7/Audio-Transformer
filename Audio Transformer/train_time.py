def train_time():
  # Loading data

  print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
  model = AudioTransformer(N_SEGMENTS, REPC_VEC_SIZE, N_ENCODERS, HIDDEN_DIM, N_HEADS, NUM_CLASSES).to(device)
  #model.load_state_dict(torch.load('my_model3_2_Transfer.pth',map_location=torch.device('cpu')))
  #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

  train_set = SpeechCommands.CustomSpeechCommandsDataset_R("../custom_speech_commands", n_segments=N_SEGMENTS, shuffle=True, vec_size=REPC_VEC_SIZE, divisor=BATCH_SIZE, control=True)
  
  train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)

  # Defining model and training options

  # Training loop
  optimizer = Adam(model.parameters(), lr=LR)
  criterion = CrossEntropyLoss()
  #scheduler = lr_scheduler.LinearLR(optimizer)
  scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

  model.train()  # Ensure model is in training mode
  for epoch in trange(EPOCHS, desc="Training"):
    epoch_start_time = time.time()
    train_loss = 0.0
    n_batches = len(train_loader)
    
    # Loop over batches
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False)):
      batch_start_time = time.time()

      # Data retrieval time (time spent getting the batch from the DataLoader)
      t0 = time.time()
      x, y = batch  # Fetch the batch
      t_data_fetch = time.time() - t0

      # Data transfer time (moving batch to device)
      t0 = time.time()
      x, y = x.to(device), y.to(device)
      t_to_device = time.time() - t0

      # Forward pass
      t0 = time.time()
      y_hat = model(x)
      t_forward = time.time() - t0

      # Loss computation
      t0 = time.time()
      loss = criterion(y_hat, y)
      t_loss = time.time() - t0

      # Backward pass
      t0 = time.time()
      optimizer.zero_grad()
      loss.backward()
      t_backward = time.time() - t0

      # Optimizer step
      t0 = time.time()
      optimizer.step()
      t_step = time.time() - t0

      batch_time = time.time() - batch_start_time

      #if batch_idx % 10 == 0:
      print(f"Epoch {epoch+1}, Batch {batch_idx}/{n_batches}: "
            f"Batch Time: {batch_time:.4f}s, "
            f"Data Fetch: {t_data_fetch:.4f}s, "
            f"To device: {t_to_device:.4f}s, "
            f"Forward: {t_forward:.4f}s, "
            f"Loss calc: {t_loss:.4f}s, "
            f"Backward: {t_backward:.4f}s, "
            f"Step: {t_step:.4f}s")

      train_loss += loss.item() * x.size(0)
      
     # End of epoch computations
    train_loss /= len(train_loader.dataset)
    scheduler.step(train_loss)
    epoch_time = time.time() - epoch_start_time
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss: {train_loss:.4f}, LR: {current_lr:.6f}, Epoch Time: {epoch_time:.2f}s")


  torch.save(model.state_dict(), f'models/ATmodel_{N_SEGMENTS}SEG_{REPC_VEC_SIZE}VEC_E{EPOCHS}_{N_HEADS}_{N_ENCODERS}_B{BATCH_SIZE}_H{HIDDEN_DIM}.pth')

