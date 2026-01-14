"""
Test different configurations to find optimal settings for license plate recognition
"""
import os
import sys
import glob
import time
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

# Enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def format_confidence(confidence):
    """Format confidence value for display"""
    if isinstance(confidence, (list, tuple)) and len(confidence) > 0:
        avg_confidence = sum(confidence) / len(confidence)
        return f"{avg_confidence:.3f}"
    elif isinstance(confidence, (int, float)):
        return f"{confidence:.3f}"
    else:
        return "N/A"


def test_configuration(config_name, use_preprocessing, use_upscaling, min_confidence, test_images):
    """Test a specific configuration"""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"  Preprocessing: {use_preprocessing}")
    print(f"  Upscaling: {use_upscaling}")
    print(f"  Min confidence: {min_confidence}")
    print(f"{'='*80}")

    try:
        # Initialize pipeline
        start_time = time.time()
        number_plate_detection_and_reading = pipeline(
            "number_plate_detection_and_reading",
            image_loader="opencv",
            upscaling=use_upscaling
        )
        init_time = time.time() - start_time

        # Process images
        start_time = time.time()
        result = number_plate_detection_and_reading(
            test_images,
            min_accuracy=min_confidence
        )
        process_time = time.time() - start_time

        (images, images_bboxs, images_points, images_zones,
         region_ids, region_names, count_lines,
         confidences, texts) = unzip(result)

        # Print results
        total_plates = 0
        for i, image_path in enumerate(test_images):
            image_name = os.path.basename(image_path)
            num_plates = len(texts[i]) if i < len(texts) else 0
            total_plates += num_plates

            print(f"\n  {image_name}:")
            if num_plates > 0:
                for j in range(num_plates):
                    plate_text = texts[i][j] if j < len(texts[i]) else "N/A"
                    region = region_names[i][j] if j < len(region_names[i]) else "N/A"
                    confidence = confidences[i][j] if j < len(confidences[i]) else []
                    conf_str = format_confidence(confidence)
                    print(f"    ✓ Plate {j+1}: {plate_text} (region: {region}, conf: {conf_str})")
            else:
                print(f"    ✗ No plates detected")

        print(f"\n  Summary:")
        print(f"    Total plates detected: {total_plates}")
        print(f"    Init time: {init_time:.2f}s")
        print(f"    Process time: {process_time:.2f}s")

        return {
            'config': config_name,
            'total_plates': total_plates,
            'init_time': init_time,
            'process_time': process_time,
            'results': texts,
            'confidences': confidences,
            'regions': region_names
        }

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return None


if __name__ == '__main__':
    # Get test images
    test_dir = './data/examples/test'
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    test_images = []

    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))

    test_images.sort()

    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {os.path.basename(img)}")

    # Test configurations
    configurations = [
        # Test 1: Baseline (no preprocessing, no upscaling)
        ("Baseline", False, False, 0.4),

        # Test 2: Lower confidence threshold
        ("Lower Confidence (0.3)", False, False, 0.3),
        ("Lower Confidence (0.2)", False, False, 0.2),

        # Test 3: Higher confidence threshold
        ("Higher Confidence (0.5)", False, False, 0.5),

        # Test 4: With upscaling
        ("With Upscaling", False, True, 0.4),
        ("Upscaling + Low Conf (0.2)", False, True, 0.2),

        # Test 5: With preprocessing
        # Note: Preprocessing requires modifying the script to apply it
        # For now we'll skip this as it requires image preprocessing
    ]

    results = []
    for config_name, preprocessing, upscaling, confidence in configurations:
        result = test_configuration(config_name, preprocessing, upscaling, confidence, test_images)
        if result:
            results.append(result)

    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} {'Plates':<10} {'Init(s)':<10} {'Process(s)':<12}")
    print(f"{'-'*80}")

    for r in results:
        print(f"{r['config']:<30} {r['total_plates']:<10} {r['init_time']:<10.2f} {r['process_time']:<12.2f}")

    # Find best configuration
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")

    max_plates = max(r['total_plates'] for r in results)
    best_configs = [r for r in results if r['total_plates'] == max_plates]

    if best_configs:
        fastest = min(best_configs, key=lambda x: x['process_time'])
        print(f"\n✓ Best configuration: {fastest['config']}")
        print(f"  - Detected {fastest['total_plates']} plates")
        print(f"  - Processing time: {fastest['process_time']:.2f}s")
