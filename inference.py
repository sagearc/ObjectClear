import os
import argparse
import torch
from .objectclear.pipelines import ObjectClearPipeline
from .objectclear.utils import resize_by_short_side
from PIL import Image
from pathlib import Path
from ..models import ObjectClearResults


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--segmentation_results', type=str, default=None,
                        help='Path to segmentation results JSON file')
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
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for inference. Default: 10')
    args = parser.parse_args()


    # ------------------------ input & output ------------------------

    IMAGE_SUFFIXES = ['.png', '.jpg', '.jpeg']

    with open(args.segmentation_results, "r") as f:
        results = ObjectClearResults.model_validate_json(f.read())

    image_path = Path(results.image_path)
    image_name = image_path.stem

    # Get successful objects directly
    successful_objects = results.get_successful_objects()

    output_dir = Path(results.output_dir) / 'objectclear'
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
    for obj in successful_objects:
        mask = Image.open(obj.mask_path).convert("L")
        resized_mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)
        masks.append(resized_mask)

    w, h = image.size

    images, attn_masks = [], []
    for i in range(0, len(masks), args.batch_size):
        print(f'Processing masks {i+1} to {min(i+args.batch_size, len(masks))} / {len(masks)}')
        batch_masks = masks[i:i+args.batch_size]
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
        images.extend(result.images)
        attn_masks.extend(result.attns)

    # -------------------- save results ---------------------    
    for obj, fused_img_pil, attn_map in enumerate(zip(obj, images, attn_masks)):            
        save_path = os.path.join(output_dir, f'removed_obj{obj.id:02d}.png')
        fused_img_pil = fused_img_pil.resize(image_or.size)
        fused_img_pil.save(save_path)

        attn_map_save_path = os.path.join(output_dir, f'attn_map{obj.id:02d}.png')
        attn_map = attn_map.resize(image_or.size)
        attn_map.save(attn_map_save_path)


    # Save ObjectClearResults as JSON
    results_path = results.output_dir / 'objectclear_results.json'
    with open(results_path, 'w') as f:
        f.write(results.model_dump_json(indent=4))


    print(f'\nAll results are saved in {output_dir}')
    print(f'ObjectClear metadata saved to: {results_path}')
