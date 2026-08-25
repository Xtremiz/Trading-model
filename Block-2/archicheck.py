import torch
import torch.nn as nn

MODEL_PATH = r"F:\Git-Hub\Trading model\models\best_cnn_model.pt"


print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint type:")
print(type(checkpoint))


# ============================================================
# CHECK CHECKPOINT
# ============================================================

if isinstance(checkpoint, dict):

    print("\nCheckpoint keys:")

    for key in checkpoint.keys():
        print("  ", key)


# ============================================================
# FIND STATE DICT
# ============================================================

state_dict = None

if isinstance(checkpoint, dict):

    possible_keys = [
        "model_state_dict",
        "state_dict",
        "model"
    ]

    for key in possible_keys:

        if key in checkpoint:

            candidate = checkpoint[key]

            if isinstance(candidate, dict):
                state_dict = candidate
                print("\nUsing state dict:", key)
                break


# ============================================================
# DIRECT STATE DICT
# ============================================================

if state_dict is None and isinstance(checkpoint, dict):

    # Check if checkpoint itself looks like a state_dict

    tensor_values = [
        value
        for value in checkpoint.values()
        if isinstance(value, torch.Tensor)
    ]

    if len(tensor_values) > 0:

        state_dict = checkpoint

        print("\nCheckpoint itself is a state_dict.")


# ============================================================
# PRINT EVERYTHING
# ============================================================

if state_dict is None:

    print("\n❌ Could not find state_dict.")

    print("\nCheckpoint contents:")

    if isinstance(checkpoint, dict):

        for key, value in checkpoint.items():

            print(
                f"{key}: "
                f"{type(value)}"
            )

    raise SystemExit


print("\n" + "=" * 70)
print("STATE DICT")
print("=" * 70)


for name, tensor in state_dict.items():

    if isinstance(tensor, torch.Tensor):

        print(
            f"{name:50s} "
            f"{tuple(tensor.shape)}"
        )

    else:

        print(
            f"{name:50s} "
            f"{type(tensor)}"
        )


# ============================================================
# DETECT LINEAR LAYERS
# ============================================================

print("\n" + "=" * 70)
print("DETECTED LINEAR LAYERS")
print("=" * 70)

linear_layers = []

for name, tensor in state_dict.items():

    if not isinstance(tensor, torch.Tensor):
        continue

    if name.endswith(".weight"):

        shape = tuple(tensor.shape)

        # Linear weight is normally [out_features, in_features]

        if len(shape) == 2:

            linear_layers.append(
                (name, shape[0], shape[1])
            )


if linear_layers:

    for i, (name, out_features, in_features) in enumerate(
        linear_layers,
        1
    ):

        print(
            f"{i}. {name}"
        )

        print(
            f"   Input neurons : {in_features}"
        )

        print(
            f"   Output neurons: {out_features}"
        )

else:

    print("No Linear layers detected.")


# ============================================================
# DETECT BATCH NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("DETECTED BATCH NORMALIZATION")
print("=" * 70)

for name, tensor in state_dict.items():

    if not isinstance(tensor, torch.Tensor):
        continue

    if "running_mean" in name:

        print(
            f"{name} -> BatchNorm features: "
            f"{tensor.shape[0]}"
        )


# ============================================================
# ARCHITECTURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ARCHITECTURE SUMMARY")
print("=" * 70)

if linear_layers:

    print(
        "\nInput features:",
        linear_layers[0][2]
    )

    print("\nNetwork:")

    for i, (_, out_features, in_features) in enumerate(
        linear_layers
    ):

        print(
            f"{in_features} -> {out_features}"
        )

    print(
        "\nOutput classes:",
        linear_layers[-1][1]
    )


# ============================================================
# TOTAL PARAMETERS
# ============================================================

total_params = 0

for tensor in state_dict.values():

    if isinstance(tensor, torch.Tensor):

        total_params += tensor.numel()


print(
    "\nTotal parameters:",
    f"{total_params:,}"
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)