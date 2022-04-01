import yaml
from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate

with open("./config.yaml", "r") as stream:
    args = yaml.safe_load(stream)

# put the train/test/infer processes into the constructor
AlectioPipeline = Pipeline(
    name=args["EXP_NAME"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
    token="1a37ef028dfa4f9197899ac7db464b61",
)

if __name__ == "__main__":
    AlectioPipeline()
