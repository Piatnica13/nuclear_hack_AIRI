#!/usr/bin/env python3
import argparse, torch, numpy as np, zstandard as zstd, json
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', default='codec_config.json')
    args = parser.parse_args()
    with open(args.config) as cf: config = json.load(cf)
    with open(args.input, 'rb') as f: compressed = f.read()
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed)
    latent = np.frombuffer(decompressed, dtype=np.uint8).reshape(config['latent_shape'])
    latent = (latent.astype(np.float32) / 255.0 * 6.0) - 3.0
    print(f"✅ Decoded latent: {latent.shape}")
if __name__ == '__main__': main()
