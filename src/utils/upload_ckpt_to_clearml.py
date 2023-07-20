from clearml import InputModel

InputModel.import_model(weights_url="/workspace/logs/train/runs/2023-07-16_23-13-17/checkpoints/old_epoch_046.ckpt",
                        name='unet-pos-weight-model',
                        project='ship-segmentation')