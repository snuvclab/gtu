
# Environment Settings
First clone this code repository
```bash
git clone --recursive [REPO_NAME(TBU)]
```

## Common Settings
Download SMPL neutral model (seems like illegal way of querying but anyway, let's use it)

```bash
# Run this under submodules/4D-Humans
wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
mkdir -p gtu/smpl_deformer/smpl/smpl_model/
mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl gtu/smpl_deformer/smpl/smpl_model/SMPL_NEUTRAL.pkl
```

If you need to use MALE or FEMALE SMPL, place similarly naming as SMPL_MALE.pkl / SMPL_FEMALE.pkl.

## Installation of main environment

Here we assume the system has CUDA11.8 and corresponding PATH settings (such as LD_LIBRARY_PATH)

```bash
# Install torch + cuda
conda create -n gtu python=3.9
conda activate gtu
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install chamfer-distance
pip install git+'https://github.com/otaheri/chamfer_distance'

# Install diff-gaussian-rasterizer (Modified to render alpha-channel (also back-propabable))
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# Install pytorch3D (It's just for debug visualization.)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

# Install xformers
pip install -U xformers==0.0.22

# Install diffusers
pip install diffusers["torch"]

# Install other dependencies
pip install -r requirements_basic.txt
```

Things to add (pytorch3d(cub,fvcore...), wandb, torch-metrics, chamfer-distance (for SMPL calculation))
Things to add future (smplx)
Not sure (ninja, setuptools)

### Possible Issues on installing diff-gaussian-rasterization 
When building diff-gaussian-rasterization, you might meet errors like 
`'glm' has not been declared`. If then, you need to install following package.

```bash
apt-get install libglm-dev
```


## Installation for preprocessing code
Humans-4D (for preprocessing the dataset)
Install dependencies for preprocessing

```bash
# Install humans 4D 
cd submodules/4D-Humans
conda create --name 4D-humans python=3.10
conda activate 4D-humans
pip install torch
pip install -e .[all]
pip install git+https://github.com/brjathu/PHALP.git
```

### Regarding issues in Humans-4D
You may see errors that torch.compiler is not detected. In this case, you need to install detectron2 with older version. (They use same v0.6 tag in recent repo, but the supported torch version is different per commits)
```bash
# Install detectron2 manually
git clone https://github.com/facebookresearch/detectron2
cd detectron2
git reset --hard 337ca34
python -m pip install -e .

```


## Install DW pose (in `gtu` conda environment)

```bash
bash -i install_dwpose.sh
```

## Install DepthAnything (in `gtu` conda environment)

```bash
cd submodules/Depth-Anything
conda activate gtu
pip install -r requirements.txt

```

## Install GroundedSAM + RAM. 
We use docker for installation here, but if you can install in conda, it's fine. 
(Current Docker build is based on CUDA 11.6 in default. be aware of it) 
(It's fine to skip it when the sequence doesn't have occlusion)

```bash
# Building Docker
export AM_I_DOCKER=True
cp GSAM_Dockerfile submodules/Grounded-Segment-Anything/Dockerfile
cd submodules/Grounded-Segment-Anything
make build-image
make run
```

### Download Required files

```bash
pip install gdown

# (If you use SAM-HQ options)
gdown 1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G # SAM_HQ_vit_l
gdown 1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8 # SAM_HQ_vit_h

mkdir -p checkpoints

# SAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mv sam_vit_h_4b8939.pth checkpoints/sam_vit_h_4b8939.pth

# GroundingDINO checkpoint (~700MiB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
mv groundingdino_swint_ogc.pth checkpoints/groundingdino_swint_ogc.pth

# RAM checkpoint (~5GiB)
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
mv ram_swin_large_14m.pth checkpoints/ram_swin_large_14m.pth
```


Also download prior for pose fitting as SPIN did.
Please refer it https://github.com/nkolot/SPIN/tree/master/data

```bash

extradata
└── spin
    ├── cube_parts.npy
    ├── gmm_08.pkl
    ├── J_regressor_extra.npy
    ├── J_regressor_h36m.npy
    ├── smpl_mean_params.npz
    ├── train.h5
    ├── vertex_texture.npy
    └── results_p1.pkl


```
The most important file among aboves is gmm_08.pkl. Else are fine even omitted.


