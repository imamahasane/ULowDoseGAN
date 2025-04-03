import time
import torch

def train_gan(generator, discriminator, loss_fn, optimizer_G, optimizer_D, dataloader, epochs, device):
    for epoch in range(epochs):
        start_time = time.time()
        
        for batch in dataloader:
            real_images = batch[0].to(device)
            
            # Generate fake images
            fake_images = generator(torch.randn_like(real_images))
            
            # Discriminator Update
            optimizer_D.zero_grad()
            real_loss = loss_fn.adversarial_loss(discriminator(real_images), True)
            fake_loss = loss_fn.adversarial_loss(discriminator(fake_images.detach()), False)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Generator Update
            optimizer_G.zero_grad()
            g_loss = loss_fn.adversarial_loss(discriminator(fake_images), True) + loss_fn.pixel_loss(fake_images, real_images)
            g_loss.backward()
            optimizer_G.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, Time: {epoch_time:.2f}s")