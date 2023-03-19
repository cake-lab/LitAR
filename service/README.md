# LitAR Service

LitAR's main codebase is written in Python, the calculation is accelerated by Numba with CUDA supports. To get the best performance, we recommend using edge devices with CUDA cores. We tested our code on a [Nvidia Jetson AGX Xavier developer kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/).

## How to Use the Code

Our Python dependencies are managed by [Pipenv](https://pipenv.pypa.io/en/latest/), please follow the [installation instructions](https://pipenv.pypa.io/en/latest/#install-pipenv-today) on its official document to install it first.

Next, activate your environment and install dependencies.

```bash
pipenv shell
pipenv install
```

Start the service.

```bash
./launch.py serve
```

## Directory Structure

- `etc`: datasets definitions and loaders.
- `litar`: LitAR system code.
- `pyreality`: a 3D computation library.
- `tests`: testing code.
- `tmp`: a local temporary file folder (DO NOT REMOVE).
- `config.py`: configuration file.
- `launch.py`: command line entry script
- `Pipfile`: pipenv dependency definition file

## Quality Settings

Please use the `configs.py` file to adjust LitAR's lighting reconstruction quality setting. We have provided the preset configurations for low, medium and high settings.
