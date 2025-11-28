#!/usr/bin/env python3
"""
Model Optimization Script
Convert PyTorch models to ONNX and apply quantization
"""

import os
import argparse
import torch
import torch.quantization
from models import GeometricMatchingModule, TryOnModuleAdvanced
from utils import load_checkpoint


def export_to_onnx(model, dummy_input, output_path, opset_version=14):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        dummy_input: Example input tensor
        output_path: Output ONNX file path
        opset_version: ONNX opset version
    """
    model.eval()
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    print(f"Model exported to ONNX: {output_path}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


def quantize_model(model, output_path):
    """
    Apply dynamic quantization to PyTorch model
    
    Args:
        model: PyTorch model
        output_path: Output path for quantized model
    """
    model.eval()
    
    # Dynamic quantization (INT8)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Quantized model saved: {output_path}")
    
    # Compare sizes
    original_size = os.path.getsize(output_path.replace('_quantized', ''))
    quantized_size = os.path.getsize(output_path)
    compression = (1 - quantized_size / original_size) * 100
    
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"Compression: {compression:.2f}%")


def convert_to_fp16(checkpoint_path, output_path):
    """
    Convert model weights to FP16 for faster inference
    
    Args:
        checkpoint_path: Input checkpoint path
        output_path: Output checkpoint path
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Convert weights to FP16
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        fp16_state_dict = {
            k: v.half() if v.dtype == torch.float32 else v
            for k, v in state_dict.items()
        }
        checkpoint['model_state_dict'] = fp16_state_dict
    
    torch.save(checkpoint, output_path)
    
    # Compare sizes
    original_size = os.path.getsize(checkpoint_path)
    fp16_size = os.path.getsize(output_path)
    compression = (1 - fp16_size / original_size) * 100
    
    print(f"FP16 model saved: {output_path}")
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"FP16 size: {fp16_size / 1024 / 1024:.2f} MB")
    print(f"Compression: {compression:.2f}%")


def benchmark_model(model, dummy_input, num_iterations=100):
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model
        dummy_input: Example input
        num_iterations: Number of iterations for benchmarking
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")


def main():
    parser = argparse.ArgumentParser(description='Optimize Vesaki-VTON models')
    parser.add_argument('--gmm_checkpoint', type=str, default='checkpoints/gmm_final.pth')
    parser.add_argument('--tom_checkpoint', type=str, default='checkpoints/tom_final.pth')
    parser.add_argument('--output_dir', type=str, default='checkpoints/optimized')
    parser.add_argument('--export_onnx', action='store_true', help='Export to ONNX')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    parser.add_argument('--fp16', action='store_true', help='Convert to FP16')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark models')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Vesaki-VTON Model Optimization")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    gmm = GeometricMatchingModule(input_nc=22, feature_dim=512, num_points=5).to(device)
    tom = TryOnModuleAdvanced(input_nc=9).to(device)
    
    load_checkpoint(args.gmm_checkpoint, gmm, device=device)
    load_checkpoint(args.tom_checkpoint, tom, device=device)
    
    # Dummy inputs
    gmm_dummy_input = torch.randn(1, 22, 1024, 768).to(device)
    garment_dummy = torch.randn(1, 3, 1024, 768).to(device)
    tom_dummy_input = (
        torch.randn(1, 3, 1024, 768).to(device),  # agnostic
        torch.randn(1, 3, 1024, 768).to(device),  # warped_garment
        torch.randn(1, 3, 1024, 768).to(device)   # person_repr
    )
    
    # Export to ONNX
    if args.export_onnx:
        print("\n" + "=" * 70)
        print("Exporting to ONNX...")
        print("=" * 70)
        
        # GMM
        gmm_onnx_path = os.path.join(args.output_dir, 'gmm.onnx')
        export_to_onnx(gmm, (gmm_dummy_input, garment_dummy), gmm_onnx_path)
        
        # TOM
        tom_onnx_path = os.path.join(args.output_dir, 'tom.onnx')
        export_to_onnx(tom, tom_dummy_input, tom_onnx_path)
    
    # Quantization
    if args.quantize:
        print("\n" + "=" * 70)
        print("Applying Quantization...")
        print("=" * 70)
        
        gmm_quant_path = os.path.join(args.output_dir, 'gmm_quantized.pth')
        tom_quant_path = os.path.join(args.output_dir, 'tom_quantized.pth')
        
        quantize_model(gmm, gmm_quant_path)
        quantize_model(tom, tom_quant_path)
    
    # FP16 conversion
    if args.fp16:
        print("\n" + "=" * 70)
        print("Converting to FP16...")
        print("=" * 70)
        
        gmm_fp16_path = os.path.join(args.output_dir, 'gmm_fp16.pth')
        tom_fp16_path = os.path.join(args.output_dir, 'tom_fp16.pth')
        
        convert_to_fp16(args.gmm_checkpoint, gmm_fp16_path)
        convert_to_fp16(args.tom_checkpoint, tom_fp16_path)
    
    # Benchmark
    if args.benchmark:
        print("\n" + "=" * 70)
        print("Benchmarking Models...")
        print("=" * 70)
        
        print("\nGMM Benchmark:")
        benchmark_model(gmm, (gmm_dummy_input, garment_dummy))
        
        print("\nTOM Benchmark:")
        benchmark_model(tom, tom_dummy_input)
    
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print(f"Optimized models saved in: {args.output_dir}")


if __name__ == '__main__':
    main()

