"""
License plate recognition on images from data/examples/test/
Outputs results in Markdown format with image links
Features auto-enhancement for low-contrast images
"""
import os
import sys
import glob
import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip


def is_low_contrast(image_path, threshold=20):
    """Check if image has low contrast (low std deviation)"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    return img.std() < threshold


def enhance_low_contrast_image(image_path, scale=3):
    """Enhance low contrast images with histogram equalization and upscaling"""
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    # Apply histogram equalization on YUV color space
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Upscale for better recognition
    h, w = enhanced.shape[:2]
    upscaled = cv2.resize(enhanced, (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)

    # Save enhanced image
    output_path = image_path.replace('.jpg', '_enhanced_auto.jpg')
    output_path = output_path.replace('.jpeg', '_enhanced_auto.jpeg')
    output_path = output_path.replace('.png', '_enhanced_auto.png')
    cv2.imwrite(output_path, upscaled)

    return output_path


def format_confidence(confidence):
    """Format confidence value for display"""
    if isinstance(confidence, (list, tuple)) and len(confidence) > 0:
        avg_confidence = sum(confidence) / len(confidence)
        return f"{avg_confidence:.3f}"
    elif isinstance(confidence, (int, float)):
        return f"{confidence:.3f}"
    else:
        return "N/A"


def generate_markdown_report(test_images, texts, region_names, confidences):
    """Generate Markdown formatted report with image links"""
    md_lines = []
    md_lines.append("# License Plate Recognition Results")
    md_lines.append("")
    md_lines.append(f"**Total images processed:** {len(test_images)}")

    total_plates = sum(len(texts[i]) for i in range(len(texts)))
    md_lines.append(f"**Total plates detected:** {total_plates}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    for i, image_path in enumerate(test_images):
        image_name = os.path.basename(image_path)
        relative_path = os.path.relpath(image_path, os.getcwd())
        num_plates = len(texts[i]) if i < len(texts) else 0

        md_lines.append(f"## Image {i+1}: [{image_name}]({relative_path})")
        md_lines.append("")
        md_lines.append(f"![{image_name}]({relative_path})")
        md_lines.append("")
        md_lines.append(f"**Detected plates:** {num_plates}")
        md_lines.append("")

        if num_plates > 0:
            md_lines.append("| # | Plate Number | Region | Confidence |")
            md_lines.append("|---|--------------|--------|------------|")

            for j in range(num_plates):
                plate_text = texts[i][j] if j < len(texts[i]) else "N/A"
                region = region_names[i][j] if j < len(region_names[i]) else "N/A"
                confidence = confidences[i][j] if j < len(confidences[i]) else []
                conf_str = format_confidence(confidence)

                md_lines.append(f"| {j+1} | **{plate_text}** | {region} | {conf_str} |")
        else:
            md_lines.append("*No plates detected*")

        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    return "\n".join(md_lines)


if __name__ == '__main__':
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    # Use GPU for processing (RTX 5070 Ti with sm_120 now supported!)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Auto-enhance low contrast images (recommended)
    # Automatically detects and enhances images with std < 20
    AUTO_ENHANCE_LOW_CONTRAST = True
    LOW_CONTRAST_THRESHOLD = 20  # std threshold for detection

    # AI Upscaling (HAT) - has issues, disabled
    USE_UPSCALING = False

    # Minimum detection confidence threshold
    MIN_DETECTION_CONFIDENCE = 0.3  # Lowered for better detection

    # Output markdown file
    OUTPUT_MD_FILE = "recognition_results.md"
    # ============================================================================

    print("Nomeroff-Net License Plate Recognition")
    print("=" * 70)
    print(f"  GPU enabled: Yes (CUDA device 0)")
    print(f"  Auto-enhance low contrast: {AUTO_ENHANCE_LOW_CONTRAST}")
    print(f"  AI upscaling (HAT): {USE_UPSCALING}")
    print(f"  Min detection confidence: {MIN_DETECTION_CONFIDENCE}")
    print(f"  Output file: {OUTPUT_MD_FILE}")
    print("=" * 70)

    # Initialize pipeline with GPU and upscaling
    number_plate_detection_and_reading = pipeline(
        "number_plate_detection_and_reading",
        image_loader="opencv",
        upscaling=USE_UPSCALING
    )

    # Get all images from test directory
    test_dir = './data/examples/test'
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    test_images = []

    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))

    if not test_images:
        print(f"\nError: No images found in {test_dir}")
        sys.exit(1)

    test_images.sort()
    print(f"\nFound {len(test_images)} images in {test_dir}\n")

    # Auto-enhance low contrast images
    processed_images = []
    enhanced_images = []

    if AUTO_ENHANCE_LOW_CONTRAST:
        print("Analyzing images for contrast enhancement...")
        for img_path in test_images:
            img = cv2.imread(img_path)
            if img is not None:
                std_dev = img.std()
                print(f"  {os.path.basename(img_path)}: std={std_dev:.1f}", end="")

                if std_dev < LOW_CONTRAST_THRESHOLD:
                    print(" -> LOW CONTRAST, enhancing...")
                    enhanced_path = enhance_low_contrast_image(img_path)
                    processed_images.append(enhanced_path)
                    enhanced_images.append(enhanced_path)
                else:
                    print(" -> OK")
                    processed_images.append(img_path)
            else:
                processed_images.append(img_path)
        print()
    else:
        processed_images = test_images

    # Process all images
    print("Running detection and recognition...")
    result = number_plate_detection_and_reading(
        processed_images,
        min_accuracy=MIN_DETECTION_CONFIDENCE
    )

    (images, images_bboxs, images_points, images_zones,
     region_ids, region_names, count_lines,
     confidences, texts) = unzip(result)

    print("Processing complete!\n")

    # Generate markdown report (using original image names)
    markdown_report = generate_markdown_report(
        test_images, texts, region_names, confidences
    )

    # Save to file
    with open(OUTPUT_MD_FILE, 'w', encoding='utf-8') as f:
        f.write(markdown_report)

    print(f"✓ Results saved to: {OUTPUT_MD_FILE}")

    # Cleanup enhanced images
    if enhanced_images:
        print(f"\nCleaning up {len(enhanced_images)} enhanced images...")
        for enhanced_path in enhanced_images:
            if os.path.exists(enhanced_path):
                os.remove(enhanced_path)
        print("Cleanup complete.")

    print()

    # Also print to console
    print(markdown_report)
