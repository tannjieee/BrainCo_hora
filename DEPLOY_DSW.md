# Aliyun DSW deployment

This project uses the existing IsaacLab image; a separate conda environment is not required.

- SSH: `ssh -p 1023 root@8.130.44.94`
- Project: `/workspace/BrainCo_hora`
- IsaacLab launcher: `/workspace/isaaclab/isaaclab.sh`
- Persistent output directory: `/mnt/nas/BrainCo_hora/outputs`
- Project `outputs` is a symbolic link to that NAS directory.
- Local `outputs`, `output`, `.git`, IDE settings, Python bytecode and credential files are excluded from deployment. Assets and grasp caches are included.

`/workspace` belongs to the container; NAS is the persistent output location. A source snapshot and deployment metadata are stored under `/mnt/nas/BrainCo_hora/deployments/` for rebuilding this workspace.

## Verification

```bash
cd /workspace/BrainCo_hora
bash scripts/dsw.sh python -m unittest discover -s tests -p test_stage1_core.py -v
bash scripts/dsw.sh python tests/check_stage1_integration.py
```

The launcher checks that `/mnt/nas` is mounted and the output symlink is correct before running. It uses Isaac Sim's bundled Python and enables the application's root-container execution option when necessary. `ISAACLAB_PATH` accepts either the installation directory (for example `/workspace/isaaclab`) or its executable `isaaclab.sh`; directory-valued image environment variables are normalized automatically.

## Strawberry Stage1

```bash
cd /workspace/BrainCo_hora
bash scripts/dsw.sh stage1 run_strawberry_s1_v2 \
  --task strawberry --num_envs 2048
```

Outputs, including TensorBoard events and checkpoints, go to:

```text
/mnt/nas/BrainCo_hora/outputs/revo3_right/run_strawberry_s1_v2/
```

Use a new output name for independent runs. Stage1 uses fixed 9.81 m/s² for strawberry by default. Setup verification is not a long training run.

## Rebuild after replacing the DSW container

Restore a deployment source snapshot into a new, empty `/workspace/BrainCo_hora` directory. Keep the existing NAS outputs. If the project has no `outputs` entry yet, create:

```bash
ln -s /mnt/nas/BrainCo_hora/outputs /workspace/BrainCo_hora/outputs
```

Do not extract a snapshot over an existing project or replace an existing `outputs` directory without checking its contents first. Set `ISAACLAB_PATH` if the new image installs IsaacLab elsewhere. Files in `/mnt/nas` are persistent storage; the container image alone does not contain them.
