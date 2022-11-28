import os
import torch
import yaml
from torch import nn, optim

from preprocessing.get_target_pitch_energy import pitch_energy_preprocess
from src.data.texts import valid_symbols
from src.model.Fastspeech2 import FastSpeech2
from src.model.Loss import Loss
from utils import Dict
import wandb
from tqdm import tqdm
from src.data import dataloader as loader


def create_wandb_log(epoch, loss, total_loss):
    return {
        "epoch": epoch,
        "mel_loss": loss[0].item(),
        "pitch_loss": loss[1].item(),
        "energy_loss": loss[2].item(),
        "duration_loss": loss[3].item(),
        "total_loss": total_loss.item(),
    }


def train(model: FastSpeech2, optimizer, loss_fn, dataloader, cfg, current_step=0):
    model.train()
    device = cfg.model.base_settings.device

    model.to(device)
    wandb.watch(model, optimizer, log="all", log_freq=10)
    with tqdm(total=cfg.train.total_step) as tqdm_bar:
        for epoch in range(cfg.train.total_step // len(dataloader)):
            for batch in dataloader:
                current_step += 1
                tqdm_bar.update(1)

                x = batch["src"].long().to(model_config.base_settings.device, non_blocking=True)
                input_length = batch["src_length"].long().to(model_config.base_settings.device, non_blocking=True)
                out_length = batch["mel_length"].long().to(model_config.base_settings.device, non_blocking=True)
                mel_target = batch["mel_target"].float().to(model_config.base_settings.device, non_blocking=True)
                duration = batch["duration"].int().to(model_config.base_settings.device, non_blocking=True)
                pitch = batch["pitch"].float().to(model_config.base_settings.device, non_blocking=True)
                energy = batch["energy"].float().to(model_config.base_settings.device, non_blocking=True)
                mel_pos = batch["mel_pos"].long().to(model_config.base_settings.device, non_blocking=True)
                src_pos = batch["src_pos"].long().to(model_config.base_settings.device, non_blocking=True)
                max_mel_len = batch["mel_max_len"]
                max_input_len = batch["max_len"]

                (out, log_duration_prediction, pitch_prediction, energy_prediction), \
                (round_duration, src_masks, mel_masks) = model(
                    x, src_pos, input_length, max_input_len,
                    mel_pos, out_length, max_mel_len, duration, energy, pitch)

                loss = loss_fn((mel_target, duration, pitch, energy),
                               (out, log_duration_prediction, pitch_prediction, energy_prediction),
                               (src_masks, mel_masks))
                total_loss = loss[0] + loss[1] + loss[2] + loss[3]
                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip_thresh)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % cfg.log.log_step == 0:
                    wandb_log = create_wandb_log(epoch, loss, total_loss)
                    wandb.log(wandb_log)
                    model.train()

                if current_step % cfg.train.save_step == 0:
                    os.makedirs(cfg.data.checkpoint_path, exist_ok=True)
                    torch.save({
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    },
                    os.path.join(cfg.data.checkpoint_path, f'checkpoint_{current_step}.pth'))


if __name__ == '__main__':
    cfg = yaml.load(open("./config/fastspeech2.yaml", "r"), Loader=yaml.FullLoader)
    cfg = Dict(cfg)
    model_config = cfg.model
    data_config = cfg.data
    train_config = cfg.train
    if cfg.preprocess.need:
        preprocess_config = yaml.load(open("./config/preprocess.yaml", "r"), Loader=yaml.FullLoader)
        preprocess_config = Dict(preprocess_config)
        pitch_energy_preprocess(preprocess_config)

    dataloader = loader.make_dataset(cfg.data.data_dir, cfg.train.batch_size, cfg)
    vocab_size = len(valid_symbols)
    num_mels = cfg.data.num_mels
    pitch_energy_stats = (cfg.data.pitch_min, cfg.data.pitch_max,
                          cfg.data.energy_min, cfg.data.energy_max)

    model = FastSpeech2(vocab_size=vocab_size,
                        max_len=model_config.base_settings.max_len,
                        n_layers=model_config.base_settings.n_layers,
                        d_model=model_config.base_settings.hidden_dim,
                        d_inner=model_config.base_settings.conv_filter_size,
                        pad_idx=data_config.pad_idx,
                        conv_block_kernel_size=model_config.base_settings.conv_kernel_size,
                        variance_kernel_size=model_config.variance_predictor.kernel_size,
                        n_heads=model_config.base_settings.attention_heads,
                        num_mels=model_config.base_settings.num_mels,
                        variance_dropout_prob=model_config.variance_predictor.dropout,
                        conv_dropout_prob=model_config.base_settings.dropout,
                        pitch_energy_stats=pitch_energy_stats,
                        device=model_config.base_settings.device)

    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                           weight_decay=cfg.train.weight_decay,
                           betas=cfg.train.betas,
                           eps=cfg.train.eps)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=cfg.train.max_lr,
                                              total_steps=cfg.train.total_step,
                                              div_factor=cfg.train.div_factor,
                                              pct_start=cfg.train.pct_start,
                                              anneal_strategy=cfg.train.anneal_strategy)

    # state_dict = torch.load("checkpoints/checkpoint_155000.pth")
    # model.load_state_dict(state_dict["model"])
    # optimizer.load_state_dict(state_dict["optimizer"])
    # for state in optimizer.state:
    #     optimizer.state[state]["exp_avg"] = \
    #         optimizer.state[state]["exp_avg"].to(model_config.base_settings.device)
    #     optimizer.state[state]["exp_avg_sq"] = \
    #         optimizer.state[state]["exp_avg_sq"].to(model_config.base_settings.device)

    with wandb.init(project=cfg.log.wandb_project_name, entity="johan_ddc_team",
                    config=cfg):
        train(model, optimizer, criterion, dataloader, cfg)
