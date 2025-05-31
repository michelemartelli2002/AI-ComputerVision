import torch
from model import alexnet, improved_alexnet, pretrained_alexnet, weak_alexnet, improved_attention_alexnet
import dataset.dataset as dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

def train_cifar10(model, epochs=5):
    train_loader = dataset.get_cifar10_train_loader(batch_size=64,
                                                    num_workers=6,
                                                    resize_w=224,
                                                    resize_h=224,
                                                    augment=True)
    test_loader = dataset.get_cifar10_test_loader(batch_size=64,
                                                  num_workers=6,
                                                  resize_w=224,
                                                  resize_h=224)

    logger = TensorBoardLogger("tb_logs", name=f"{model.name}CIFAR10")  # Save logs to tb_logs/

    # --------------------------------------
    # Train
    # --------------------------------------
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=False          # We don't want checkpoints
    )

    trainer.fit(model, train_loader, test_loader)
    out_file = f"model/saved/{model.name.lower()}_trained{epochs}_cifar10.pth"
    torch.save(model.state_dict(), out_file)
    print(f"Training complete. Model saved as {out_file}")

def train_cifar100(model, epochs=5):
    train_loader = dataset.get_cifar100_train_loader(batch_size=64,
                                                     num_workers=6,
                                                     resize_w=224,
                                                     resize_h=224,
                                                     augment=True)
    test_loader = dataset.get_cifar100_test_loader(batch_size=64,
                                                   num_workers=6,
                                                   resize_w=224,
                                                   resize_h=224)

    logger = TensorBoardLogger("tb_logs", name=f"{model.name}CIFAR100")  # Save logs to tb_logs/

    # --------------------------------------
    # Train
    # --------------------------------------
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=False          # We don't want checkpoints
    )

    trainer.fit(model, train_loader, test_loader)
    out_file = f"model/saved/{model.name.lower()}_trained{epochs}_cifar100.pth"
    torch.save(model.state_dict(), out_file)
    print(f"Training complete. Model saved as {out_file}")

def train_catbreeds67(model, epochs=5):
    train_loader = dataset.get_CatBreeds67_train_loader(batch_size=64,
                                                        num_workers=6,
                                                        resize_w=224,
                                                        resize_h=224,
                                                        augment=True)
    test_loader = dataset.get_CatBreeds67_test_loader(batch_size=64,
                                                      num_workers=6,
                                                      resize_w=224,
                                                      resize_h=224)

    logger = TensorBoardLogger("tb_logs", name=f"{model.name}CATBREEDS67")  # Save logs to tb_logs/

    # --------------------------------------
    # Train
    # --------------------------------------
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=False          # We don't want checkpoints
    )

    trainer.fit(model, train_loader, test_loader)
    out_file = f"model/saved/{model.name.lower()}_trained{epochs}_catbreeds67.pth"
    torch.save(model.state_dict(), out_file)
    print(f"Training complete. Model saved as {out_file}")

def train_flowers102(model, epochs=5):
    train_loader = dataset.get_flowers102_train_loader(batch_size=64,
                                                       num_workers=6,
                                                       resize_w=224,
                                                       resize_h=224,
                                                       augment=True)
    test_loader = dataset.get_flowers102_val_loader(batch_size=64,
                                                    num_workers=6,
                                                    resize_w=224,
                                                    resize_h=224)

    logger = TensorBoardLogger("tb_logs", name=f"{model.name}Flowers102")  # Save logs to tb_logs/

    # --------------------------------------
    # Train
    # --------------------------------------
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=False,          # We don't want checkpoints
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, test_loader)
    out_file = f"model/saved/{model.name.lower()}_trained{epochs}_flowers102.pth"
    torch.save(model.state_dict(), out_file)
    print(f"Training complete. Model saved as {out_file}")

def train_flowers102(model, epochs=5):
    train_loader = dataset.get_oxfordIIITpet_train_loader(batch_size=64,
                                                          num_workers=6,
                                                          resize_w=224,
                                                          resize_h=224,
                                                          augment=True)
    test_loader = dataset.get_oxfordIIITpet_test_loader(batch_size=64,
                                                       num_workers=6,
                                                       resize_w=224,
                                                       resize_h=224)

    logger = TensorBoardLogger("tb_logs", name=f"{model.name}Oxford37")  # Save logs to tb_logs/

    # --------------------------------------
    # Train
    # --------------------------------------
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=False,          # We don't want checkpoints
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, test_loader)
    out_file = f"model/saved/{model.name.lower()}_trained{epochs}_oxford37.pth"
    torch.save(model.state_dict(), out_file)
    print(f"Training complete. Model saved as {out_file}")

def train_caltech101(model, epochs=5):
    train_loader = dataset.get_caltech101_train_loader(batch_size=64,
                                                          num_workers=6,
                                                          resize_w=224,
                                                          resize_h=224,
                                                          augment=True)
    val_loader = dataset.get_caltech101_val_loader(batch_size=64,
                                                       num_workers=6,
                                                       resize_w=224,
                                                       resize_h=224)

    logger = TensorBoardLogger("tb_logs", name=f"{model.name}Caltech101")  # Save logs to tb_logs/

    # --------------------------------------
    # Train
    # --------------------------------------
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=False,          # We don't want checkpoints
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)
    out_file = f"model/saved/{model.name.lower()}_trained{epochs}_caltech101.pth"
    torch.save(model.state_dict(), out_file)
    print(f"Training complete. Model saved as {out_file}")

def main():
    model = pretrained_alexnet.get_caltech101_model()
    train_caltech101(model=model,
                     epochs=50)

if __name__ == "__main__":
    main()
