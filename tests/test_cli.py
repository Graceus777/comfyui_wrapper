from comfyui_wrapper.cli import build_parser


def test_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["generate", "--prompt", "a cat", "--steps", "20"])
    assert args.cmd == "generate"
    assert args.prompt == "a cat"
    assert args.steps == 20


def test_parser_anima_lllite_control():
    parser = build_parser()
    args = parser.parse_args([
        "generate",
        "--anima-control-image", "depth.png",
        "--anima-lllite-model", "anima-lllite-depth-1.safetensors",
        "--anima-control-strength", "0.8",
    ])
    assert args.anima_control_image == "depth.png"
    assert args.anima_lllite_model == "anima-lllite-depth-1.safetensors"
    assert args.anima_control_strength == 0.8


def test_parser_prep_slots():
    parser = build_parser()
    args = parser.parse_args([
        "prep", "--zone", "all_char", "--prompt-config", "sd_hair",
        "--slot", "pose=standing,kneeling", "--out", "jobs.json",
    ])
    assert args.zone == "all_char"
    assert args.slot == ["pose=standing,kneeling"]


def test_parser_video():
    parser = build_parser()
    args = parser.parse_args(["video", "--image", "still.png", "--prompt", "snow", "--frames", "124"])
    assert args.cmd == "video"
    assert args.image == "still.png"
    assert args.frames == 124


def test_parser_video_recipe():
    parser = build_parser()
    args = parser.parse_args([
        "video", "--image", "still.png", "--recipe", "acc",
        "--acc-file", "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
    ])
    assert args.recipe == "acc"
    assert args.acc_file.endswith("Acc-8Step.safetensors")


def test_parser_video_upscale_flags():
    parser = build_parser()
    args = parser.parse_args([
        "video", "--image", "still.png", "--recipe", "native",
        "--latent-upscale", "--rtx-vsr",
    ])
    assert args.recipe == "native"
    assert args.latent_upscale is True
    assert args.rtx_vsr is True


def test_parser_batch_jobs():
    parser = build_parser()
    args = parser.parse_args(["batch", "--jobs", "configs/jobs/run.json", "--resume"])
    assert args.jobs.endswith("run.json")
    assert args.resume is True
    assert args.skip_exists is None


def test_parser_skip_exists_override():
    parser = build_parser()
    assert parser.parse_args(["prep", "--zone", "z", "--skip-exists"]).skip_exists is True
    assert parser.parse_args(["prep", "--zone", "z", "--no-skip-exists"]).skip_exists is False
