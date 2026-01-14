"""
Analyze problematic image in detail
"""
import os
import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

# Enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def analyze_image(image_path, min_confidence=0.1):
    """Analyze image with very low confidence threshold"""
    print(f"\nAnalyzing: {os.path.basename(image_path)}")
    print("="*80)

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image!")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    print(f"Image shape: {img.shape}")

    # Check image statistics
    print(f"\nImage statistics:")
    print(f"  Mean: {img.mean():.2f}")
    print(f"  Std: {img.std():.2f}")
    print(f"  Min: {img.min()}")
    print(f"  Max: {img.max()}")

    # Try detection with very low threshold
    print(f"\nTrying detection with confidence threshold: {min_confidence}")

    pipeline_obj = pipeline(
        "number_plate_detection_and_reading",
        image_loader="opencv",
        upscaling=False
    )

    result = pipeline_obj([image_path], min_accuracy=min_confidence)
    (images, images_bboxs, images_points, images_zones,
     region_ids, region_names, count_lines,
     confidences, texts) = unzip(result)

    # Print bbox info
    if images_bboxs and len(images_bboxs[0]) > 0:
        print(f"\nDetected {len(images_bboxs[0])} bounding boxes:")
        for i, bbox in enumerate(images_bboxs[0]):
            print(f"  Box {i+1}: {bbox}")

    # Print detection results
    if texts and len(texts[0]) > 0:
        print(f"\nDetected {len(texts[0])} plates:")
        for i in range(len(texts[0])):
            plate = texts[0][i]
            region = region_names[0][i] if i < len(region_names[0]) else "N/A"
            conf = confidences[0][i] if i < len(confidences[0]) else []

            if isinstance(conf, (list, tuple)) and len(conf) > 0:
                avg_conf = sum(conf) / len(conf)
                conf_str = f"{avg_conf:.3f}"
            else:
                conf_str = "N/A"

            print(f"  {i+1}. {plate} (region: {region}, conf: {conf_str})")
    else:
        print(f"\n✗ No plates detected even with confidence={min_confidence}")


    # Save annotated image
    if images_bboxs and len(images_bboxs[0]) > 0:
        annotated = img.copy()
        for bbox in images_bboxs[0]:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        output_path = image_path.replace('.jpg', '_annotated.jpg')
        cv2.imwrite(output_path, annotated)
        print(f"\n✓ Saved annotated image to: {output_path}")


if __name__ == '__main__':
    print("Detailed Image Analysis")
    print("="*80)

    # Analyze problematic image with different upscaling
    test_images = [
        './data/examples/test/0a6c9609d3ecb0e24ad66296716d1412.jpg',
        './data/examples/test/0a4cf1ab47d219cc61225f599a6427f9.jpg',
    ]

    for img_path in test_images:
        # Test with original
        analyze_image(img_path, min_confidence=0.1)

        # Test with 2x upscaling
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        upscaled = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_LANCZOS4)
        upscaled_path = img_path.replace('.jpg', '_3x.jpg')
        cv2.imwrite(upscaled_path, upscaled)

        print(f"\n--- Testing with 3x upscaling ---")
        analyze_image(upscaled_path, min_confidence=0.1)

        # Cleanup
        if os.path.exists(upscaled_path):
            os.remove(upscaled_path)

        print("\n" + "="*80 + "\n")
