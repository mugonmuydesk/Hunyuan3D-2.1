"""
RunPod Serverless Handler for Hunyuan3D-2.1 (Image-to-3D)

Generates high-fidelity 3D models with PBR materials from input images.
"""

import os
import sys
import base64
import tempfile
from pathlib import Path

# Add model paths
sys.path.insert(0, '/app/hy3dshape')
sys.path.insert(0, '/app/hy3dpaint')

import runpod


# Global pipelines (loaded once)
shape_pipeline = None
paint_pipeline = None


def load_pipelines():
    """Load the shape and paint pipelines."""
    global shape_pipeline, paint_pipeline

    if shape_pipeline is None:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        print("Loading shape pipeline...")
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            cache_dir='/models'
        )
        print("Shape pipeline loaded.")

    if paint_pipeline is None:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        print("Loading paint pipeline...")
        config = Hunyuan3DPaintConfig(
            max_num_view=int(os.environ.get('MAX_NUM_VIEW', 6)),
            resolution=int(os.environ.get('TEXTURE_RESOLUTION', 512))
        )
        paint_pipeline = Hunyuan3DPaintPipeline(config)
        print("Paint pipeline loaded.")

    return shape_pipeline, paint_pipeline


def save_base64_to_file(b64_data: str, output_path: str) -> str:
    """Decode base64 data and save to file."""
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]
    decoded = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(decoded)
    return output_path


def encode_file_to_base64(file_path: str) -> str:
    """Read file and encode to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for Hunyuan3D-2.1.

    Input:
        image_base64: Base64 encoded input image (required)
        generate_texture: Whether to generate PBR textures (default: true)
        output_format: Output format - 'glb' or 'obj' (default: 'glb')
        num_views: Number of views for texture generation (default: 6)
        texture_resolution: Texture resolution (default: 512)

    Output:
        model: Base64 encoded 3D model file (GLB or OBJ)
        format: The output format used
    """
    job_input = job["input"]

    # Validate required inputs
    if "image_base64" not in job_input:
        return {"error": "Missing required field: image_base64"}

    # Extract parameters
    image_b64 = job_input["image_base64"]
    generate_texture = job_input.get("generate_texture", True)
    output_format = job_input.get("output_format", "glb").lower()
    num_views = job_input.get("num_views", 6)
    texture_resolution = job_input.get("texture_resolution", 512)

    if output_format not in ["glb", "obj"]:
        return {"error": f"Invalid output_format: {output_format}. Use 'glb' or 'obj'."}

    # Load pipelines
    try:
        shape_pipe, paint_pipe = load_pipelines()
    except Exception as e:
        return {"error": f"Failed to load pipelines: {e}"}

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save input image
        image_ext = ".png"
        if "image/jpeg" in image_b64 or "image/jpg" in image_b64:
            image_ext = ".jpg"
        image_path = temp_path / f"input{image_ext}"

        try:
            save_base64_to_file(image_b64, str(image_path))
        except Exception as e:
            return {"error": f"Failed to decode input image: {e}"}

        # Generate shape (untextured mesh)
        try:
            print(f"Generating shape from image...")
            mesh = shape_pipe(image=str(image_path))[0]
            print("Shape generation complete.")
        except Exception as e:
            return {"error": f"Shape generation failed: {e}"}

        # Generate texture if requested
        if generate_texture:
            try:
                print(f"Generating textures ({num_views} views, {texture_resolution}px)...")
                # Update paint pipeline config if needed
                paint_pipe.config.max_num_view = num_views
                paint_pipe.config.resolution = texture_resolution
                mesh = paint_pipe(mesh, image_path=str(image_path))
                print("Texture generation complete.")
            except Exception as e:
                return {"error": f"Texture generation failed: {e}"}

        # Export mesh
        output_path = temp_path / f"output.{output_format}"
        try:
            if output_format == "glb":
                mesh.export(str(output_path))
            else:
                mesh.export(str(output_path), file_type='obj')
            print(f"Exported to {output_format.upper()}")
        except Exception as e:
            return {"error": f"Failed to export mesh: {e}"}

        # Encode output
        try:
            model_b64 = encode_file_to_base64(str(output_path))
        except Exception as e:
            return {"error": f"Failed to encode output: {e}"}

        return {
            "model": model_b64,
            "format": output_format
        }


# For local testing
if __name__ == "__main__":
    test_job = {
        "input": {
            "image_base64": "...",
            "generate_texture": True,
            "output_format": "glb"
        }
    }
    print(handler(test_job))
else:
    runpod.serverless.start({"handler": handler})
