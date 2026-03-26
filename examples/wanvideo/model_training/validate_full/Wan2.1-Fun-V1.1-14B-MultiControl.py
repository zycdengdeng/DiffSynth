import torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# Load pipeline with multi-control checkpoint
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # Use the initialized multi-control checkpoint (in_dim=80)
        ModelConfig(model_id="models/Wan2.1-Fun-V1.1-14B-MultiControl", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
)

# Load trained weights
state_dict = load_state_dict("models/train/Wan2.1-Fun-V1.1-14B-MultiControl_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict)

# Load three control videos
sparse_depth = [VideoData("data/multi_control_dataset/test_depth.mp4", height=480, width=832)[i] for i in range(81)]
sparse_color = [VideoData("data/multi_control_dataset/test_color.mp4", height=480, width=832)[i] for i in range(81)]
bbox_video = [VideoData("data/multi_control_dataset/test_bbox.mp4", height=480, width=832)[i] for i in range(81)]
reference_image = VideoData("data/multi_control_dataset/test_ref.mp4", height=480, width=832)[0]

# Generate with multi-control
video = pipe(
    prompt="a street scene with vehicles and buildings",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    control_video_sparse_depth=sparse_depth,
    control_video_sparse_color=sparse_color,
    control_video_bbox=bbox_video,
    reference_image=reference_image,
    seed=1, tiled=True,
)
save_video(video, "video_Wan2.1-Fun-V1.1-14B-MultiControl.mp4", fps=15, quality=5)
