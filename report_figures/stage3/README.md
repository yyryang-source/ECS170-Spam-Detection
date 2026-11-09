# Stage 3 Experiment Results

| Dataset | Selected model | Test images | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [MNIST](mnist/experiment_summary.json) | Kernel 3×3 residual | 10,000 | 99.33% | 99.34% | 99.33% | 99.33% |
| [ORL](orl/experiment_summary.json) | Kernel 3×3 residual | 40 | 97.50% | 96.25% | 97.50% | 96.67% |
| [CIFAR-10](cifar10/experiment_summary.json) | Kernel 5×5 residual | 10,000 | 84.95% | 85.12% | 84.95% | 84.88% |

## Dataset and run provenance

The ORL experiment used the instructor ZIP and the [template entry point](../../script/stage_3_script/script_cnn_orl.py). A stratified split allocated 270 training images to model fitting and 90 to validation. The instructor's 40 test images were held out. The two residual variants tied on validation macro F1, so the 3×3 variant, which was listed first, was selected.

MNIST was trained directly on the instructor ZIP using the [template entry point](../../script/stage_3_script/script_cnn_mnist.py). All three configurations trained for three epochs on the same stratified split of 54,000 fit images and 6,000 validation images. The instructor's 10,000 test images were held out and used to score the selected configuration.

CIFAR-10 was trained using the [standalone entry point](../../script/stage3/run_stage3.py) with 15 epochs, batch size 128, seed 42, and cached torchvision data. The cached images, labels and record order matched the supplied ZIP for both partitions. The saved checkpoint reproduced its validation and test metrics through the template loader on the instructor ZIP. [CIFAR-10 verification](cifar10/checkpoint_verification.json)

Every experiment selected checkpoints and architectures on validation data. The linked summaries include validation ablations, hyperparameters, split counts, and the course archive SHA-256. Each dataset directory also includes epoch loss histories for all three configurations.

## Architecture and convergence

![CNN architecture](architecture.png)

![Training and validation loss by epoch for MNIST, ORL, and CIFAR-10](convergence.png)
