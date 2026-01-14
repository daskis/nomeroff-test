"""
Fix low contrast images with aggressive enhancement
"""
import os
import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

# Enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def enhance_contrast(image_path, method='adaptive'):
    """Apply aggressive contrast enhancement"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    if method == 'histogram_eq':
        # Convert to YUV
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    elif method == 'adaptive':
        # CLAHE on LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l = clahe.apply(l)

        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    elif method == 'normalize':
        # Normalize to full range
        enhanced = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    elif method == 'gamma':
        # Gamma correction
        gamma = 0.5  # Brighten
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(img, table)

    elif method == 'combined':
        # Normalize first
        enhanced = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Then CLAHE
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Sharpen
        kernel = np.array([[0, -1, 0],
                          [-1, 5,-1],
                          [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    else:
        enhanced = img

    return enhanced


def test_enhancement(image_path, method_name, enhancement_method, scale=3):
    """Test enhancement method"""
    print(f"\nTesting: {method_name}")
    print("-"*80)

    # Load and enhance
    img = cv2.imread(image_path)
    enhanced = enhance_contrast(image_path, method=enhancement_method)

    if enhanced is None:
        print("Failed to enhance image")
        return 0

    # Upscale
    h, w = enhanced.shape[:2]
    upscaled = cv2.resize(enhanced, (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)

    # Save for testing
    temp_path = image_path.replace('.jpg', f'_temp_{enhancement_method}.jpg')
    cv2.imwrite(temp_path, upscaled)

    # Check statistics
    print(f"  Enhanced stats: mean={upscaled.mean():.1f}, std={upscaled.std():.1f}, min={upscaled.min()}, max={upscaled.max()}")

    # Try recognition
    try:
        pipeline_obj = pipeline(
            "number_plate_detection_and_reading",
            image_loader="opencv",
            upscaling=False
        )

        result = pipeline_obj([temp_path], min_accuracy=0.1)
        (_, images_bboxs, _, _, _, region_names, _, confidences, texts) = unzip(result)

        if texts and len(texts[0]) > 0:
            print(f"  ✓ Detected {len(texts[0])} plates:")
            for i in range(len(texts[0])):
                plate = texts[0][i]
                region = region_names[0][i] if i < len(region_names[0]) else "N/A"
                conf = confidences[0][i] if i < len(confidences[0]) else []

                if isinstance(conf, (list, tuple)) and len(conf) > 0:
                    avg_conf = sum(conf) / len(conf)
                    conf_str = f"{avg_conf:.3f}"
                else:
                    conf_str = "N/A"

                print(f"      {plate} (region: {region}, conf: {conf_str})")

            # Save best result
            if len(texts[0]) > 0:
                best_path = image_path.replace('.jpg', f'_BEST_{enhancement_method}.jpg')
                cv2.imwrite(best_path, upscaled)
                print(f"  ✓ Saved best result to: {os.path.basename(best_path)}")

            os.remove(temp_path)
            return len(texts[0])
        else:
            print(f"  ✗ No plates detected")
            os.remove(temp_path)
            return 0

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return 0


if __name__ == '__main__':
    print("Aggressive Contrast Enhancement Testing")
    print("="*80)

    problem_image = './data/examples/test/0a6c9609d3ecb0e24ad66296716d1412.jpg'

    # Show original stats
    img = cv2.imread(problem_image)
    h, w = img.shape[:2]
    print(f"\nOriginal image: {w}x{h}")
    print(f"  Stats: mean={img.mean():.1f}, std={img.std():.1f}, min={img.min()}, max={img.max()}")

    # Test different enhancement methods
    methods = [
        ('Histogram Equalization', 'histogram_eq'),
        ('Adaptive CLAHE', 'adaptive'),
        ('Normalize', 'normalize'),
        ('Gamma Correction', 'gamma'),
        ('Combined (Normalize + CLAHE + Sharpen)', 'combined'),
    ]

    results = {}
    for method_name, method_key in methods:
        count = test_enhancement(problem_image, method_name, method_key, scale=3)
        results[method_name] = count

    # Summary
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<50} {'Plates Detected':<20}")
    print(f"{'-'*80}")

    for method, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"{status} {method:<48} {count:<20}")

    # Find best
    best_method = max(results.items(), key=lambda x: x[1])
    if best_method[1] > 0:
        print(f"\n✓ BEST METHOD: {best_method[0]} (detected {best_method[1]} plates)")
    else:
        print(f"\n✗ No method successfully detected plates on this image")
        print(f"   The image may be too degraded or may not contain a readable license plate")
