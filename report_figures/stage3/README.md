# Stage 3 Experiment Results

| Dataset | Selected model | Test images | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [MNIST](mnist/experiment_summary.json) | Kernel 3×3 residual | 10,000 | 99.33% | 99.34% | 99.33% | 99.33% |
| [ORL](orl/experiment_summary.json) | Kernel 3×3 residual | 40 | 97.50% | 96.25% | 97.50% | 96.67% |
| [CIFAR-10](cifar10/experiment_summary.json) | Kernel 5×5 residual | 10,000 | 71.21% | 73.38% | 71.21% | 70.54% |

## Dataset and run provenance

The ORL experiment used the instructor ZIP and the [template entry point](../../script/stage_3_script/script_cnn_orl.py). A stratified split allocated 270 training images to model fitting and 90 to validation. The instructor's 40 test images were held out. The two residual variants tied on validation macro F1, so the 3×3 variant, which was listed first, was selected.

MNIST was trained directly on the instructor ZIP using the [template entry point](../../script/stage_3_script/script_cnn_mnist.py). All three configurations trained for three epochs on the same stratified split of 54,000 fit images and 6,000 validation images. The instructor's 10,000 test images were held out and used to score the selected configuration.

The supplemental CIFAR-10 experiment trained all three configurations directly on the instructor ZIP for three epochs using the [template entry point](../../script/stage_3_script/script_cnn_cifar.py). The configurations shared a stratified split of 45,000 fit images and 5,000 validation images. The selected configuration was scored on the instructor's 10,000 test images.

Every experiment selected checkpoints and architectures on validation data. The linked summaries include validation ablations, hyperparameters, split counts, and the course archive SHA-256. Each dataset directory also includes epoch loss histories for all three configurations.

## Architecture and convergence

![CNN architecture](architecture.png)

![Training and validation loss by epoch for MNIST, ORL, and CIFAR-10](convergence.png)
