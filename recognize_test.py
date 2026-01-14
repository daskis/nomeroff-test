"""
License plate recognition on images from data/examples/test/
Outputs results in Markdown format with image links
"""
import os
import sys
import glob
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip


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

    # Image preprocessing (CLAHE, denoise, sharpen)
    # WARNING: Can WORSEN results on good quality images!
    # Only enable for very low contrast/noisy images
    USE_PREPROCESSING = False

    # AI Upscaling (HAT) - has issues, disabled
    # Automatically applied for plates < 50px height
    USE_UPSCALING = False

    # Minimum detection confidence threshold
    MIN_DETECTION_CONFIDENCE = 0.4

    # Output markdown file
    OUTPUT_MD_FILE = "recognition_results.md"
    # ============================================================================

    print("Nomeroff-Net License Plate Recognition")
    print("=" * 70)
    print(f"  GPU enabled: Yes (CUDA device 0)")
    print(f"  Image preprocessing: {USE_PREPROCESSING}")
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

    # Process all images
    print("Running detection and recognition...")
    result = number_plate_detection_and_reading(
        test_images,
        min_accuracy=MIN_DETECTION_CONFIDENCE
    )

    (images, images_bboxs, images_points, images_zones,
     region_ids, region_names, count_lines,
     confidences, texts) = unzip(result)

    print("Processing complete!\n")

    # Generate markdown report
    markdown_report = generate_markdown_report(
        test_images, texts, region_names, confidences
    )

    # Save to file
    with open(OUTPUT_MD_FILE, 'w', encoding='utf-8') as f:
        f.write(markdown_report)

    print(f"✓ Results saved to: {OUTPUT_MD_FILE}")
    print()

    # Also print to console
    print(markdown_report)
