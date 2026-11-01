"""
Damage Severity Estimation Engine
Inspired by PotholeGrade-BD methodology.
Computes polygonal contours, bounding boxes, scale categorization (small/medium/large),
photometric depth proxy, road surface damage ratio, and civil engineering maintenance recommendations.
"""
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple


class DamageSeverityEngine:
    """
    Evaluates segmented road damage regions to quantify severity, categorize scale,
    and generate actionable maintenance intelligence.
    """

    def __init__(self, small_thresh: int = 1000, medium_thresh: int = 5000):
        self.small_thresh = small_thresh
        self.medium_thresh = medium_thresh

    def assess(self, image_rgb: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Assess road damage from RGB image and predicted probability mask.

        Args:
            image_rgb: RGB image as uint8 numpy array (H, W, 3).
            mask: Probability mask as float array (H, W) in [0, 1] or binary uint8.
            threshold: Probability threshold for binary binarization.

        Returns:
            Dictionary containing severity score, status, instances list, scale counts, and civil recommendations.
        """
        h, w = mask.shape[:2]
        total_image_pixels = h * w

        # Binarize mask
        if mask.dtype != np.uint8:
            binary_mask = (mask >= threshold).astype(np.uint8) * 255
        else:
            binary_mask = mask

        # Grayscale image for photometric shadow/depth analysis
        if len(image_rgb.shape) == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_rgb

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        instances: List[Dict[str, Any]] = []
        small_count = 0
        med_count = 0
        large_count = 0
        total_damaged_pixels = int(np.sum(binary_mask > 0))
        area_ratio = total_damaged_pixels / float(total_image_pixels)

        # Surrounding road luminance baseline (excluding damaged regions)
        road_pixels = gray[binary_mask == 0]
        ambient_luminance = float(np.mean(road_pixels)) if len(road_pixels) > 0 else 128.0

        max_depth_proxy = 0.0

        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            if area < 30:  # Filter minor noise artifacts
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Polygonal approximation for smooth polygon rendering
            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            polygon_pts = approx.reshape(-1, 2).tolist()

            # Scale classification
            if area < self.small_thresh:
                scale_cat = "small"
                small_count += 1
            elif area <= self.medium_thresh:
                scale_cat = "medium"
                med_count += 1
            else:
                scale_cat = "large"
                large_count += 1

            # Photometric Depth Proxy:
            # Pothole cavity accumulates shadow & gravel depression => lower intensity & higher edge gradient
            pothole_crop_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted_cnt = cnt - np.array([x, y])
            cv2.drawContours(pothole_crop_mask, [shifted_cnt], -1, 255, -1)
            
            pothole_patch = gray[y : y + bh, x : x + bw]
            patch_pixels = pothole_patch[pothole_crop_mask == 255]

            if len(patch_pixels) > 0:
                pothole_mean_lum = float(np.mean(patch_pixels))
                # Intensity deficit relative to surrounding pavement
                lum_deficit = max(0.0, ambient_luminance - pothole_mean_lum) / (ambient_luminance + 1e-5)
                # Texture / edge roughness proxy using Laplacian variance
                laplacian_var = float(cv2.Laplacian(pothole_patch, cv2.CV_64F).var())
                roughness_score = min(1.0, laplacian_var / 500.0)
                # Combined depth proxy [0.0 - 1.0]
                depth_proxy = min(1.0, 0.65 * lum_deficit + 0.35 * roughness_score)
            else:
                depth_proxy = 0.2

            if depth_proxy > max_depth_proxy:
                max_depth_proxy = depth_proxy

            # Confidence score (mean probability in contour region)
            if mask.dtype != np.uint8:
                prob_patch = mask[y : y + bh, x : x + bw]
                conf = float(np.mean(prob_patch[pothole_crop_mask == 255])) if len(prob_patch[pothole_crop_mask == 255]) > 0 else 0.85
            else:
                conf = 0.90

            instances.append({
                "id": idx + 1,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "polygon": polygon_pts,
                "area_px": int(area),
                "scale": scale_cat,
                "confidence": round(conf, 3),
                "depth_proxy": round(depth_proxy, 3),
            })

        # Calculate Overall Damage Severity Index (DSI) [0 - 100]
        # PotholeGrade formulation: weighted combination of area ratio, depth proxy, and crater count
        damage_percent = area_ratio * 100.0
        pothole_count = len(instances)
        count_factor = min(1.0, pothole_count / 5.0)

        raw_severity = (
            55.0 * min(1.0, damage_percent / 12.0) +
            30.0 * max_depth_proxy +
            15.0 * count_factor
        )
        severity_score = int(np.clip(raw_severity, 0, 100)) if pothole_count > 0 else 0

        # Classify Severity Level
        if severity_score == 0:
            severity_level = "None"
            badge_color = "emerald"
            recommendation = "Road pavement is in optimal condition. No intervention required."
            action_code = "SAFE"
        elif severity_score < 25:
            severity_level = "Low"
            badge_color = "blue"
            recommendation = "Surface raveling or shallow micro-pothole detected. Schedule routine crack sealing and surface rejuvenation during regular maintenance cycle."
            action_code = "SEAL_COAT"
        elif severity_score < 50:
            severity_level = "Moderate"
            badge_color = "amber"
            recommendation = "Moderate localized pothole detected. Pavement structure beginning to fail. Deploy cold-mix asphalt patch within 14 days to prevent water ingress."
            action_code = "COLD_PATCH"
        elif severity_score < 75:
            severity_level = "Severe"
            badge_color = "orange"
            recommendation = "Severe deep road defect posing immediate hazard to vehicular tires, suspension, and cyclist safety. High-priority hot-mix asphalt patching required within 48 hours."
            action_code = "HOT_MIX_REPAIR"
        else:
            severity_level = "Critical"
            badge_color = "rose"
            recommendation = "Critical road failure / cratering network. Base course subgrade compromised. Immediate lane closure and full-depth pavement reconstruction recommended."
            action_code = "EMERGENCY_RECONSTRUCTION"

        return {
            "pothole_count": pothole_count,
            "total_damaged_pixels": total_damaged_pixels,
            "damaged_area_percentage": round(damage_percent, 2),
            "severity_score": severity_score,
            "severity_level": severity_level,
            "badge_color": badge_color,
            "action_code": action_code,
            "recommendation": recommendation,
            "scale_distribution": {
                "small": small_count,
                "medium": med_count,
                "large": large_count,
            },
            "max_depth_proxy": round(max_depth_proxy, 3),
            "instances": instances,
        }
