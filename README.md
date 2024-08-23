# OpenMPL

[![arXiv](https://img.shields.io/badge/arXiv-2408.10805-<COLOR>.svg)](https://arxiv.org/abs/2408.10805)
>**[MPL: Lifting 3D Human Pose from Multi-view 2D Poses, T-CAP ECCV24](https://arxiv.org/abs/2408.10805)**
>
>Seyed Abolfazl Ghasemzadeh, Alexandre Alahi, Christophe De Vleeschouwer
>
>[*arxiv 2408.10805*](https://arxiv.org/abs/2408.10805)
>
>
## Installation

To get started with OpenMPL, follow these steps:

### Steps

1. **Clone the Repository:**

    ```bash
    git clone git@github.com:aghasemzadeh/OpenMPL.git
    cd OpenMPL
    ```

2. **Install Dependencies:**


    ```bash
    pip install -r requirements.txt
    ```

    Install amass framework through the following link to be able to run MHP:

    https://github.com/nghorbani/amass

    Install mmpose for running off-the-shelf 2D pose estimation:

    https://github.com/open-mmlab/mmpose


3. **Datasets Setup:**

    You will need AMASS dataset for training (+ camera calibrations from your test dataset) and a dataset for testing in MPL. We test our framework on two datasets: CMU and Human3.6M.

    #### CMU

    For CMU data, please follow [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) to prepare images and annotations.
    Then run the command below after correcting the paths:

    ```bash
    cd MPL/data
    sh preprocess_cmu_panoptic_all_cams.sh
    ```

    #### Human3.6M

    For Human36M data, please follow [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to prepare images and annotations.
    Then run the command below after correcting the paths:

    ```bash
    cd MPL/data
    sh preprocess_h36m.sh
    ```

    #### AMASS

    For running MHP (generating 3D dataset for MPL), you need to download the amass dataset. Please follow [AMASS](https://amass.is.tue.mpg.de/index.html) for downloading the pose SMPL data. For running MHP, you will need to install [amass](https://github.com/nghorbani/amass) framework and follow their guidelines.

    After correcting the paths, you can run the command below:

    ```bash
    cd MPH
    sh run_mmpose_00_cmu_calibs.sh      # for CMU
    sh run_mmpose_00_h36m_calibs.sh     # for Human3.6M
    ```

    Remember to go through the code in these files to be sure you run everything. There are some parallel scripts that you need to take care yourself. (Some are commented)

## Citation

Please make sure to cite our paper if you utilize our code.

```
@ARTICLE{Ghasemzadeh2024-ln,
  title         = "{MPL}: Lifting {3D} Human Pose from Multi-view {2D} Poses",
  author        = "Ghasemzadeh, Seyed Abolfazl and Alahi, Alexandre and De
                   Vleeschouwer, Christophe",
  month         =  aug,
  year          =  2024,
  copyright     = "http://creativecommons.org/licenses/by-nc-sa/4.0/",
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  eprint        = "2408.10805"
}
```

## Acknowledgement

Part of our code is borrowed from [PPT](https://github.com/HowieMa/PPT/tree/main) and [PoseFormer](https://github.com/zczcwh/PoseFormer/tree/main). We thank the authors for releasing the codes.
    
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Happy coding! If you have any questions, feel free to open an issue or contact us directly.
