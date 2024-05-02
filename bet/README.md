## Setup

Setup the environment of bet and then install lang-sam and segment anything from source
```
cd bet/models/lang-segment-anything
pip install -e .

cd bet/models/segment-anything
pip install -e .
```

Hard code the trajectory dataset length in RelayKitchenMultiviewTrajectoryDataset class.

Then train the model
```
python3 train.py --config-name=train_kitchen
```