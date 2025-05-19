# UGD


[UGD](...) is a simple offline reinforcement learning plugin algorithm. It generates data in the data sparse area through the uncertainty guided diffusion model to expand the data range of the offline data set. The effect of the algorithm has been verified on multiple algorithms and datasets.

Here is presented the [UGD framework implemented based on the CQL algorithm](...).

## Installation

1. Install and use the included Ananconda environment
```
$ conda env create -f environment.yml
$ source activate SimpleCQL
```

2. Add this repo directory to your `PYTHONPATH` environment variable.
```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run Experiments

You can run CQL experiments using the following command:
```
python -m SimpleCQL.conservative_sac_main \
    --env 'halfcheetah-medium-v0' \
    --logging.output_dir './experiment_output'
```

If you want to run on CPU only, just add the `--device='cpu'` option.


## Visualize Experiments
You can visualize the experiment metrics with viskit:
```
python -m viskit './experiment_output'
```
and simply navigate to [http://localhost:5000/](http://localhost:5000/)



