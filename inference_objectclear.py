import os
import argparse
import glob
import torch
from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side
from PIL import Image
import numpy as np
from pathlib import Path



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/imgs', 
                        help='Input image or folder. Default: inputs/imgs')
    parser.add_argument('-m', '--mask_path', type=str, default='./inputs/masks',
                        help='Input mask image or folder. Default: inputs/masks')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
                        help='Output folder. Default: results/<input_name>')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="Path to cache directory")
    parser.add_argument('--use_fp16', action='store_true', 
                        help='Use float16 for inference')
    parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed for torch.Generator. Default: 42')
    parser.add_argument('--steps', type=int, default=20, 
                        help='Number of diffusion inference steps. Default: 20')
    parser.add_argument('--guidance_scale', type=float, default=2.5, 
                        help='CFG guidance scale. Default: 2.5')
    parser.add_argument('--no_agf', action='store_true', 
                        help='Disable Attention Guided Fusion')
    args = parser.parse_args()
    
    
    # ------------------------ input & output ------------------------
    IMAGE_SUFFIXES = ['.png', '.jpg', '.jpeg']

    image_path = Path(args.input_path)
    image_name = image_path.stem

    mask_paths = []
    for f in Path(args.mask_path).iterdir():
        if f.suffix.lower() in IMAGE_SUFFIXES:
            mask_paths.append(f)
    mask_paths.sort()

    if args.output_path is not None:
        output_dir = Path(args.output_path)
    else:
        output_dir = Path('output') / image_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------ set up ObjectClear pipeline -------------------
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    variant = "fp16" if args.use_fp16 else None
    generator = torch.Generator(device=device).manual_seed(args.seed)
    use_agf = not args.no_agf
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=torch_dtype,
        apply_attention_guided_fusion=use_agf,
        cache_dir=args.cache_dir,
        variant=variant,
    )
    pipe.to(device)
    
    
    # -------------------- start to processing ---------------------
    print(f'Processing: {image_path}')
    
    image = Image.open(image_path).convert("RGB")
    image_or = image.copy()

    # Our model was trained on 512×512 resolution.
    # Resizing the input so that the **shorter side is 512** helps achieve the best performance.
    image = resize_by_short_side(image, 512, resample=Image.BICUBIC)

    masks = []
    for p in mask_paths:
        mask = Image.open(p).convert("L")
        resized_mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)
        masks.append(resized_mask)

    w, h = image.size

    result = pipe.batch_inference(
        prompt="remove the instance of object",
        image=image,
        mask_images=masks,
        generator=generator,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=h,
        width=w,
        return_attn_map=True,
    )
    
    for i, (fused_img_pil, mask_img) in enumerate(zip(result.images, result.attns)):
        # save results
        save_path = os.path.join(output_dir, f'{image_name}_removed_obj{i+1}.png')
        fused_img_pil = fused_img_pil.resize(image_or.size)
        fused_img_pil.save(save_path)

        mask_img = mask_img.resize(image_or.size)
        mask_save_path = os.path.join(output_dir, f'{image_name}_attn_map{i+1}.png')
        mask_img.save(mask_save_path)

    print(f'\nAll results are saved in {output_dir}')
