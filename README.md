# Hunyuan3D-2.1 RunPod Serverless

Generate high-fidelity 3D models with PBR (Physically Based Rendering) materials from images using Tencent's Hunyuan3D-2.1 model.

## Features

- **Image-to-3D Generation**: Create detailed 3D models from single images
- **PBR Materials**: Production-ready textures with physically accurate rendering
- **Multiple Output Formats**: GLB and OBJ support
- **Configurable Quality**: Adjust texture resolution and view count

## API Reference

### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_base64` | string | required | Base64 encoded input image |
| `generate_texture` | boolean | true | Whether to generate PBR textures |
| `output_format` | string | "glb" | Output format: "glb" or "obj" |
| `num_views` | integer | 6 | Number of views for texture generation |
| `texture_resolution` | integer | 512 | Texture resolution in pixels |

### Output

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Base64 encoded 3D model file |
| `format` | string | Output format used (glb/obj) |

### Example Request

```json
{
  "input": {
    "image_base64": "<base64-encoded-image>",
    "generate_texture": true,
    "output_format": "glb",
    "num_views": 6,
    "texture_resolution": 512
  }
}
```

## Requirements

- **VRAM**: ~29GB (10GB shape + 21GB texture)
- **Recommended GPUs**: A100 40GB/80GB, A40, L40S, RTX A6000

## Credits

Based on [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) by Tencent.

## License

Apache 2.0 (see original repository for details)
