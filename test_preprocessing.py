"""
Test preprocessing techniques for low-resolution images
"""
import os
import cv2
import glob
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

# Enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def upscale_image(image_path, scale=4, method='LANCZOS'):
    """Upscale image using different methods"""
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    h, w = img.shape[:2]
    new_size = (w * scale, h * scale)

    # Choose interpolation method
    methods = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'LANCZOS': cv2.INTER_LANCZOS4,
    }

    interpolation = methods.get(method, cv2.INTER_LANCZOS4)
    upscaled = cv2.resize(img, new_size, interpolation=interpolation)

    # Save upscaled image
    output_path = image_path.replace('.jpg', f'_upscaled_{scale}x_{method}.jpg')
    cv2.imwrite(output_path, upscaled)

    return output_path


def enhance_image(image_path):
    """Apply CLAHE enhancement"""
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    output_path = image_path.replace('.jpg', '_enhanced.jpg')
    cv2.imwrite(output_path, sharpened)

    return output_path


def test_preprocessing_method(name, images, min_confidence=0.3):
    """Test a preprocessing method"""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    try:
        pipeline_obj = pipeline(
            "number_plate_detection_and_reading",
            image_loader="opencv",
            upscaling=False
        )

        result = pipeline_obj(images, min_accuracy=min_confidence)
        (_, _, _, _, _, region_names, _, confidences, texts) = unzip(result)

        total_plates = 0
        for i, image_path in enumerate(images):
            image_name = os.path.basename(image_path)
            num_plates = len(texts[i]) if i < len(texts) else 0
            total_plates += num_plates

            print(f"\n  {image_name}:")
            if num_plates > 0:
                for j in range(num_plates):
                    plate_text = texts[i][j] if j < len(texts[i]) else "N/A"
                    region = region_names[i][j] if j < len(region_names[i]) else "N/A"
                    confidence = confidences[i][j] if j < len(confidences[i]) else []

                    if isinstance(confidence, (list, tuple)) and len(confidence) > 0:
                        avg_conf = sum(confidence) / len(confidence)
                        conf_str = f"{avg_conf:.3f}"
                    else:
                        conf_str = "N/A"

                    print(f"    ✓ {plate_text} (region: {region}, conf: {conf_str})")
            else:
                print(f"    ✗ No plates detected")

        print(f"\n  Total: {total_plates} plates")
        return total_plates

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return 0


if __name__ == '__main__':
    import numpy as np

    # Get original test images
    test_dir = './data/examples/test'
    test_images = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))
    # Filter out previously processed images
    test_images = [img for img in test_images if 'upscaled' not in img and 'enhanced' not in img]

    print("Original images:")
    for img in test_images:
        img_data = cv2.imread(img)
        h, w = img_data.shape[:2]
        print(f"  - {os.path.basename(img)}: {w}x{h}")

    results = {}

    # Test 1: Original images
    results['Original'] = test_preprocessing_method("Original (no preprocessing)", test_images)

    # Test 2: Upscaled 2x with different methods
    for method in ['LINEAR', 'CUBIC', 'LANCZOS']:
        upscaled_images = []
        for img in test_images:
            upscaled_path = upscale_image(img, scale=2, method=method)
            upscaled_images.append(upscaled_path)

        results[f'Upscaled 2x ({method})'] = test_preprocessing_method(
            f"Upscaled 2x ({method})",
            upscaled_images
        )

    # Test 3: Upscaled 4x LANCZOS
    upscaled_images_4x = []
    for img in test_images:
        upscaled_path = upscale_image(img, scale=4, method='LANCZOS')
        upscaled_images_4x.append(upscaled_path)

    results['Upscaled 4x (LANCZOS)'] = test_preprocessing_method(
        "Upscaled 4x (LANCZOS)",
        upscaled_images_4x
    )

    # Test 4: Upscaled 4x + Enhancement
    enhanced_images = []
    for img in upscaled_images_4x:
        enhanced_path = enhance_image(img)
        enhanced_images.append(enhanced_path)

    results['Upscaled 4x + Enhanced'] = test_preprocessing_method(
        "Upscaled 4x + Enhanced (CLAHE + Sharpen)",
        enhanced_images
    )

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<40} {'Plates Detected':<20}")
    print(f"{'-'*80}")

    for method, count in results.items():
        print(f"{method:<40} {count:<20}")

    # Cleanup temporary files
    print(f"\n\nCleaning up temporary files...")
    temp_images = glob.glob(os.path.join(test_dir, '*upscaled*.jpg'))
    temp_images.extend(glob.glob(os.path.join(test_dir, '*enhanced*.jpg')))
    for temp in temp_images:
        os.remove(temp)
    print(f"Removed {len(temp_images)} temporary files")
