"""
Initialize a multi-control Wan2.1-Fun-V1.1-14B model from the single-control checkpoint.

This script expands patch_embedding from in_dim=48 to in_dim=80:
  - Original: noise(16) + control(16) + y(16) = 48
  - New:      noise(16) + sparse_depth(16) + sparse_color(16) + bbox(16) + y(16) = 80

The first 48 channels of patch_embedding reuse pretrained weights.
The new 32 channels (for two additional controls) are initialized with small random values.
"""
import torch
import argparse
from safetensors.torch import load_file, save_file
from diffsynth import hash_state_dict_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to original PAI/Wan2.1-Fun-V1.1-14B-Control safetensors file(s), comma-separated if multiple")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the multi-control checkpoint")
    parser.add_argument("--init_std", type=float, default=0.02,
                        help="Std for random initialization of new control channels")
    args = parser.parse_args()

    # Load original state dict
    input_paths = args.input_path.split(",")
    state_dict = {}
    for path in input_paths:
        state_dict.update(load_file(path.strip()))

    # Expand patch_embedding.weight: (5120, 48, 1, 2, 2) -> (5120, 80, 1, 2, 2)
    key_weight = "patch_embedding.weight"
    key_bias = "patch_embedding.bias"

    old_weight = state_dict[key_weight]  # (out_dim, 48, 1, 2, 2)
    out_dim = old_weight.shape[0]
    old_in_dim = old_weight.shape[1]  # 48
    new_in_dim = 80
    extra_channels = new_in_dim - old_in_dim  # 32

    print(f"Expanding {key_weight}: ({out_dim}, {old_in_dim}, ...) -> ({out_dim}, {new_in_dim}, ...)")

    # New channels initialized with small random values
    new_channels = torch.randn(
        out_dim, extra_channels, *old_weight.shape[2:],
        dtype=old_weight.dtype
    ) * args.init_std

    # Insert new control channels after the original control channel (position 32)
    # Original layout: noise(0:16) + control(16:32) + y(32:48)
    # New layout:      noise(0:16) + ctrl1(16:32) + ctrl2(32:48) + ctrl3(48:64) + y(64:80)
    # So we keep noise(0:16) + original_control(16:32), add 32 new channels, then y(32:48)
    new_weight = torch.cat([
        old_weight[:, :32, ...],    # noise(16) + original_control(16)
        new_channels,                # two new controls(32)
        old_weight[:, 32:, ...],    # y(16)
    ], dim=1)

    state_dict[key_weight] = new_weight
    print(f"  Result shape: {new_weight.shape}")

    # Print the new model hash for model_configs.py
    model_hash = hash_state_dict_keys(state_dict)
    print(f"\nModel hash (update model_configs.py with this): {model_hash}")

    # Save
    save_file(state_dict, args.output_path)
    print(f"Saved multi-control checkpoint to: {args.output_path}")


if __name__ == "__main__":
    main()
