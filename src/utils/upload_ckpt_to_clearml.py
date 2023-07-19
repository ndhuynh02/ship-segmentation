from clearml import Task

# initialize a task
task = Task.init(project_name='ship-segmentation',
                 task_name='upload_checkpoint')

# add and upload local file artifact
task.upload_artifact("unet_checkpoint",
                     artifact_object='ckpts/unet-pos_weight-epoch_046.ckpt'
                     )
