

Get images and train VAE:

set `TRAIN = True` in /models/dip_vae.py and experiment.py

```
python video2img.py
select validation set and put images in /dataset/val
python run.py
```



Generate z-latent from DIP-VAE:

set `TRAIN = False` in /models/dip_vae.py and experiment.py

set `ckpt_path` in predict.py (line 44)

```
python move_imgs.py (reorganize the dataset)
python predict_multiple_demos.py
```



Get the dataset and use hiss to train the model:

```
cd hiss/dataset
python copy_demos.py
python copy_z_latent.py

cd ..
conda activate vt_state
python ./data_processing/process_box_insertion.py
python create_dataset.py --config-name box_insertion_tactile.yaml
python train.py --config-name box_insertion_hiss_config_tactile data_env.task.modality=tactile model=lstm
```



Get the validation set z-latent inference:

```
python predict.py --config-name box_insertion_hiss_config_tactile  (Change the model path on line 68)

```



Reconstruct images:

```
cd DipVAE
conda activate vt_state
python z_decoder.py (change the z_latent_file_path and output_dir)
python visualize_results.py (change the file_path)
python visualize_gt.py (add groundtruth images)
```