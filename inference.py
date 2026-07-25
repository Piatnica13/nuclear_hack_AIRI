#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import zstandard as zstd
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input .bin file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default='codec_config.json')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    # Decompress
    with open(args.input, 'rb') as f:
        compressed = f.read()
    
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed)
    
    # Convert to numpy
    latent = np.frombuffer(decompressed, dtype=np.uint8)
    latent = latent.reshape(config['latent_shape'])
    
    # Dequantize
    latent = (latent.astype(np.float32) / 255.0 * 6.0) - 3.0
    
    print(f"✅ Decoded latent: {latent.shape}")

if __name__ == '__main__':
    main()
