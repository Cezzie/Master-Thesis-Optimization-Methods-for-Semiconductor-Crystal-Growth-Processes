# Optimization Methods for Semiconductor Crystal Growth Processes

This repository contains code and resources related to my Master’s thesis on the optimization of semiconductor crystal growth using numerical simulations and machine learning techniques.

By J.V.J. Schardijn

## License

This project is licensed under the **GNU General Public License v3.0**.
For full license terms, see the [LICENSE](LICENSE) file.

## Dependencies

The project relies on the following open-source libraries:

* [**PyTorch**](https://pytorch.org/) — for deep learning and optimization (BSD-3-Clause License)
* [**Matplotlib**](https://matplotlib.org/) — for data visualization and animations (PSF License)
* [**scikit-fem**](https://github.com/kinnala/scikit-fem) — for finite element method (FEM) simulations (BSD-3-Clause License)

## Installation

You can create a conda environment using the provided `requirements.txt` file:

```bash
conda create --name crystal-growth-env python=3.10
conda activate crystal-growth-env
pip install -r requirements.txt
```

> 📝 If you're working in Jupyter, make sure to install the kernel:

```bash
python -m ipykernel install --user --name crystal-growth-env --display-name "Python (crystal-growth)"
```

## Usage Notes

* The **notebook `0. Heat and Stress Solver.ipynb`** is easy to use and can be run independently. It provides a good entry point for exploring the simulation components.

* For the **training and optimization scripts**, make sure:

  * You are **logged into Weights & Biases (wandb)**. You can log in by running:

    ```bash
    wandb login
    ```
  * 📁 Ensure that the `trajectories/` folder exists and is properly populated, as the scripts depend on input data
