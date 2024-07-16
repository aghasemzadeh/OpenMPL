# OpenMPL

## Installation

To get started with [Project Name], follow these steps:

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


    You can change your prefered configurations in the sh file
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Happy coding! If you have any questions, feel free to open an issue or contact us directly.
