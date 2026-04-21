"""
Task 4: Explainable AI (XAI) to provide visual transparency for the main Administrator on Task 2 Image classifier.
Provides 10 distict Visualisation methods:
1. Grad-Cam: Uses gradient flowing into the last conv layer to produce heatmap of where the model looked for its decision.
2. LIME: Breaks image into superpixels and randomly hides them to learn which regions mattered the most.
3. Integrated Gradients: Accumulates gradients from a baseline to the input to compute pixel-level contribution.
4. SHAP: Uses blur masking to assign each region's exact contribution to the prediction.
5. Occlusion: Slides a grey patch over the image and measures confidence drop to find regions that break the prediction when hidden.
6. Counterfactual Explanation: Breaks images into meaningful zones, restore each zone one by one and see which one improves scores the most, replace a zone with the average color of the healthy parts of the fruits.
7. SmoothGrad: Adds noise to the input multiple times and averages gradients to reduce noisy saliency maps and highlight stable signals.
8. Eigen-CAM: Uses main patterns in the model's feature map to show most important parts.
9. Score-CAM: Avoids gradients entirely by masking with activation maps and measuring output scores, making it suitable when gradients are noisy.
10. Feature Ablation: Removes superpixel regions and observes prediction changes to identify which areas are critical.
"""

import copy
import cv2
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms
from typing import Dict, Tuple, List

# XAI Library Imports
from captum.attr import IntegratedGradients, Occlusion, FeatureAblation, visualization as viz
from lime import lime_image
from pytorch_grad_cam import GradCAM, EigenCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries, slic
import shap
        
       

# Task 2 
from task2_model import build_model, _extract_checkpoint_state_dict, CONFIG


# Wrapper Models: Grad-CAM expects a single output tensor, since our Task 2 model outputs two (logits, quality_scores), we create wrappers to seperate them.

class ClassifierWrapper(torch.nn.Module):
    """Isolates the classification head for gradient-based XAI."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits, _ = self.model(x)
        return logits
    
class QualityWrapper(torch.nn.Module):
    """Isolated the Quality head."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        _, quality_scores = self.model(x)
        return quality_scores
    
class QualityTarget:
    """Targets specific quality regression indices (Colour, Size, Ripeness)."""
    def __init__(self, target_index: int):
        self.target_index = target_index
    def __call__(self, model_output):
        # Return the specific quality score (0: colour, 1: size, 2: ripeness) (handles if batch is stripped or not)
        return model_output[self.target_index] if len(model_output.shape) == 1 else model_output[:, self.target_index]
    

class ProduceXAI:
    def __init__(self, model_path: str | Path, device: str = None): 
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initialising XAI Module on device: {self.device}")
    
        # Load checkpoint to get class names
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", ["Fresh", "Rotten"])
        num_classes = len(self.class_names)

        # Build model
        self.model = build_model(num_classes=num_classes, device=self.device, use_pretrained=False)
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval() 

        # Preprocessing
        self.image_size = checkpoint.get("image_size", 224)
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(), # Converts to tensor: scales pixel values from 0.0 to 1.0
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Numbers as recomended by docs https://docs.pytorch.org/vision/0.9/models.html computed over the ImageNet dataset. 
        ])

        # Last convolutional layer
        self.target_layers = [self.model.backbone.features[-1]]

    # Helper Functions
    def _convert_rgb_and_resize(self, image_path: str | Path, use_scaling: bool = False, get_resized: bool = False, return_image: bool = False) -> Image.Image:
        """
        Opens image from image path, converts to RGB, resizes it and tranforms it into an array.
        
        Arguments:
        - image_path (Required): the path to the image.
        - use_scaling (optional): bool to use scaling (/255)
        - get_resized (optional): bool to get resized image
        - return_image (optional): bool to return the converted image (not resized)
        """
        image = Image.open(image_path).convert("RGB")

        if use_scaling and get_resized and return_image:
            print("You cannot set any 2 of use_scaling get_resized and return_image to True. It will only return the image.")
            pass

        if return_image:
            return image
        
        img_resized = image.resize((self.image_size, self.image_size))

        if get_resized:
            return img_resized
        
        img_np = np.array(img_resized)

        if use_scaling:
            scaled_np = img_np / 255
            return scaled_np
        
        return img_np

    def _get_input_tensor(self, img: Image) -> torch.Tensor:
        input_tensor = self.preprocess(img).unsqueeze(0).to(self.device) # Adds batch dimension as torch sometimes requires it
        return input_tensor
    
    def _get_model_outputs(self, image_path: str | Path) -> Tuple[str, float, Dict[str, float]]:
        """Runs the image through task 2 model and gets the scores"""
        img_pil = self._convert_rgb_and_resize(image_path, return_image=True)
        input_tensor = self._get_input_tensor(img_pil)
        self.model.eval()
        with torch.no_grad():
            logits, quality_tensor = self.model(input_tensor)
            
            # Process Classification
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            label = self.class_names[idx.item()]
            
            # Process Quality Scores (Colour, Size, Ripeness)
            # Remove batch dimension and convert to numpy
            scores = quality_tensor.squeeze(0).cpu().numpy()
            
        quality_dict = {
            "colour": round(float(scores[0]), 2),
            "size": round(float(scores[1]), 2),
            "ripeness": round(float(scores[2]), 2)
        }
        
        return label, float(conf.item()), quality_dict

    def _convert_matplot_to_img(self, fig, plt = None) -> Image.Image:
        buf = io.BytesIO() # Save to disk
        fig.savefig(buf, format='jpeg', bbox_inches='tight') # save plot into the buffer
        buf.seek(0) # move pointer back to start of buffer
        pil_image = Image.open(buf)
        if plt:
            plt.close(fig)
        return pil_image

    def generate_gradcam_explanations(self, image_path: str | Path) -> Dict[str, Image.Image]:
        """
        Generates 4 heatmaps explaining:
        1. Classification
        2. Colour
        3. Size
        4. Ripeness

        Returns a dictionary of PIL Images.
        """
        
        img = self._convert_rgb_and_resize(image_path, return_image=True)
        rgb_img = self._convert_rgb_and_resize(image_path, use_scaling=True)
        input_tensor = self._get_input_tensor(img)

        explanations = {}
        # Explain Classification (Fresh vs Rotten)
        cls_wrapper = ClassifierWrapper(self.model).to(self.device)
        with GradCAM(model=cls_wrapper, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :] # [0, :] removes batch dimension, targets=None means it will target the highest scoring class
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True) # Overlay heatmap on image
            explanations["classification"] = Image.fromarray(cam_image)

        # 2. Explain Quality Scores (Colour, size, ripeness)
        quality_metrics = {"colour": 0, "size": 1, "ripeness": 2}
        quality_wrapper = QualityWrapper(self.model).to(self.device)

        with GradCAM(model=quality_wrapper, target_layers=self.target_layers) as cam:
            for metric_name, metric_idx in quality_metrics.items():
                target = [QualityTarget(target_index=metric_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                explanations[metric_name] = Image.fromarray(cam_image)

        return explanations
    
    def generate_lime_explanations(self, image_path: str | Path, num_samples = 500) -> Tuple[Image.Image, List[Tuple[int, float]]]:
        """
        Uses LIME to explain the Fresh/Rotten classification by breaking the image into superpixels and testing them.
        """         
        img_np = self._convert_rgb_and_resize(image_path)

        def batch_predict(images_numpy):
            self.model.eval()
            batch = torch.stack([self.preprocess(Image.fromarray(img)) for img in images_numpy], dim=0).to(self.device)

            with torch.no_grad():
                logits, _ = self.model(batch)
                probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()
        
        explainer = lime_image.LimeImageExplainer()

        print("Generating LIME Explanation (this may take 10-30 seconds)...")
        explanation = explainer.explain_instance(
            img_np,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples # Lower samples = faster but less accurate
        )

        # Get mask for top predicted class
        top_class = explanation.top_labels[0]
        # This returns a list of (segment_id, weight)
        feature_weights = explanation.local_exp[top_class]
        temp, mask = explanation.get_image_and_mask(
            top_class,
            positive_only=True, # Only show areas that contributed to the prediction
            num_features=5, # Show top 5 superpixels
            hide_rest=True
        )

        # Draw green boundaries around important areas
        img_boundary = mark_boundaries(temp / 255.0, mask)
        # Convert back to PIL image
        img_boundary_uint8 = np.uint8(img_boundary * 255)

         # Draw numbers on the regions 
        segments = explanation.segments
        for i in range(np.max(segments) + 1):
            # Find the center of this segment to place the text
            coords = np.argwhere(segments == i)
            if len(coords) == 0: continue
            cy, cx = coords.mean(axis=0).astype(int)
            cv2.putText(img_boundary_uint8, str(i), (cx+1, cy+1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(img_boundary_uint8, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        return Image.fromarray(img_boundary_uint8), feature_weights
    
    def generate_integrated_gradient(self, image_path: str | Path, target_idx: int = None) -> Image.Image:
        """Uses integrated Gradients to provide pixel-level contribution"""
        img = self._convert_rgb_and_resize(image_path, return_image=True)
        img_np = self._convert_rgb_and_resize(image_path)

        input_tensor = self._get_input_tensor(img)
        input_tensor.requires_grad = True

        cls_wrapper = ClassifierWrapper(self.model).to(self.device)
        cls_wrapper.eval()
        ig = IntegratedGradients(cls_wrapper)

        # Determine target class (if None use models highest predicted class)
        if target_idx is None:
            with torch.no_grad():
                logits = cls_wrapper(input_tensor)
                target_idx = torch.argmax(logits, dim=1).item()

        attributions, approximation_error = ig.attribute(input_tensor, target=target_idx,  return_convergence_delta=True)

        # Convert from [1, 3, 224, 224] -> [224, 224, 3]
        attributions_np = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        fig, axis = viz.visualize_image_attr(
            attributions_np,
            img_np,
            method="alpha_scaling",
            sign="absolute_value", # Highlights magnitude of importance (both positive and negative)
            show_colorbar=True,
            title=f"Integrated Gradients (Class: {self.class_names[target_idx]})",
            use_pyplot=False
        )

        ig_pil_image = self._convert_matplot_to_img(fig)
        return ig_pil_image

    
    def generate_shap_explanation(self, image_path: str | Path, n_evals=300) -> Image.Image:
        """
        Uses SHAP to calculate Shapley values for the image.
        This explains exactly how much each region shifted the probability.
        """
        
        img_np = self._convert_rgb_and_resize(image_path)

        # SHAP expects a function that takes [Batch, H, W, C] and returns [Batch, Classes]
        def predict_fn(images_np):
            self.model.eval()
            batch_tensors = []
            for img in images_np:
                pil_img = Image.fromarray(img.astype(np.uint8))
                batch_tensors.append(self.preprocess(pil_img))
            
            batch = torch.stack(batch_tensors, dim=0).to(self.device)
            with torch.no_grad():
                logits, _ = self.model(batch)
                probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()

        # Use a blur masker to hide parts of the image during testing
        masker = shap.maskers.Image("blur(32,32)", (self.image_size, self.image_size, 3))
        
        # Partition explainer is much faster for images than pixel-by-pixel SHAP
        explainer = shap.Explainer(predict_fn, masker, output_names=self.class_names)

        print(f"Generating SHAP Explanation (Running {n_evals} evaluations)...")
    
        shap_values = explainer(
            img_np[np.newaxis, ...], 
            max_evals=n_evals, 
            batch_size=50,
            outputs=shap.Explanation.argsort.flip[:1] # Focus on the top predicted class
        )

        # shap.image_plot doesn't return a figure directly, so we capture the current plot
        plt.clf() # Clear any existing plots
        shap.image_plot(shap_values, show=False)
        fig = plt.gcf()
        shap_pil_image = self._convert_matplot_to_img(fig, plt)
        return shap_pil_image


    def generate_occlusion_explanation(self, image_path: str | Path, target_idx: int = None) -> Image.Image:
        """
        Uses Occlusion (The 'Box Test') to see which regions cause the model's confidence to drop when they are hidden.
        """
        
        img = self._convert_rgb_and_resize(image_path, return_image=True)
        img_np = self._convert_rgb_and_resize(image_path)

        input_tensor = self._get_input_tensor(img)

        cls_wrapper = ClassifierWrapper(self.model).to(self.device)
        occ = Occlusion(cls_wrapper)

        if target_idx is None:
            with torch.no_grad():
                logits = cls_wrapper(input_tensor)
                target_idx = torch.argmax(logits, dim=1).item()

        print("Generating Occlusion Explanation (The Box Test)...")
        # sliding_window_shapes: The size of the grey box (15x15 pixels)
        # strides: How many pixels the box jumps (8 pixels)
        attributions = occ.attribute(
            input_tensor,
            strides=(3, 8, 8),
            target=target_idx,
            sliding_window_shapes=(3, 15, 15),
            baselines=0 # The color of the box (0 = black/grey)
        )

        attributions_np = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        fig, axis = viz.visualize_image_attr(
            attributions_np,
            img_np,
            method="blended_heat_map",
            sign="positive", # Only show where hiding the spot decreased confidence
            show_colorbar=True,
            title=f"Occlusion Sensitivity (Class: {self.class_names[target_idx]})",
            use_pyplot=False
        )

        occ_pil_image = self._convert_matplot_to_img(fig, plt)
        return occ_pil_image


    def generate_counterfactual(self, image_path: str | Path) -> Tuple[Image.Image, str]:
        """
        Identifies the deal breaker region that is preventing the image 
        from being classified as 'Fresh'. 
        """

        img_np = self._convert_rgb_and_resize(image_path)
        
        # Segment image into 20 meaningful zones
        segments = slic(img_np, n_segments=20, compactness=10, sigma=1)
        
        # Dynamically find the "Healthy" target for this specific fruit
        current_label, _, _ = self._get_model_outputs(image_path)
        fruit_type = current_label.split('__')[0]
        try:
            # Find index that contains the fruit name AND the word "healthy"
            target_healthy_idx = [i for i, s in enumerate(self.class_names) 
                                 if fruit_type in s and 'healthy' in s.lower()][0]
        except IndexError:
            target_healthy_idx = 0 # Fallback

        # Get the baseline (What does the AI think now?)
        def get_fresh_score(image_array):
            self.model.eval()
            pil_img = Image.fromarray(image_array.astype(np.uint8))
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = self.model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                fresh_idx = self.class_names.index("fresh") if "fresh" in self.class_names else 0
                return probs[0][fresh_idx].item()

        initial_fresh_score = get_fresh_score(img_np)
        
        # 4. Define Healing color (Median color of the fruit, ignoring background)
        fruit_mask = img_np.mean(axis=2) < 250 
        median_color = np.median(img_np[fruit_mask], axis=0) if np.any(fruit_mask) else [200, 200, 200]

        # Restore each segment one by one and see which one improves the score most
        best_improvement = 0
        dealbreaker_segment = -1
        
        for i in range(np.max(segments) + 1):
            temp_img = copy.deepcopy(img_np)
            temp_img[segments == i] = median_color
            
            new_score = get_fresh_score(temp_img)
            improvement = new_score - initial_fresh_score
            
            if improvement > best_improvement:
                best_improvement = improvement
                dealbreaker_segment = i

        # 6. Visualization
        mask = np.zeros(segments.shape, dtype=bool)
        if dealbreaker_segment != -1:
            mask[segments == dealbreaker_segment] = True
        
        overlay = img_np.copy()
        overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)
        
        result_img = label2rgb(mask, overlay, colors=[(0, 255, 0)], alpha=0.3, bg_label=0)
        result_np = (result_img * 255).astype(np.uint8)
        
        # Draw "HERE" label on the image
        if dealbreaker_segment != -1:
            coords = np.argwhere(mask == True)
            if len(coords) > 0:
                cy, cx = coords.mean(axis=0).astype(int)
                cv2.putText(result_np, "HERE", (cx - 35, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        explanation_text = (
            f"The highlighted green region is the 'Decision Anchor'.\n"
            f"If this area were repaired, confidence in '{self.class_names[target_healthy_idx]}' "
            f"would rise by {best_improvement*100:.1f}%."
        )

        return Image.fromarray(result_np), explanation_text
    
    def generate_smoothgrad(self, image_path: str | Path, n_samples=20, stdev_spread=0.15) -> Image.Image:
        """
        Generates a SmoothGrad map. Averages multiple noisy saliency maps to remove noise.
        """

        img = Image.open(image_path).convert("RGB")
        input_tensor = self._get_input_tensor(img)
        
        # Calculate standard deviation for noise based on the range of pixels
        stdev = stdev_spread * (input_tensor.max() - input_tensor.min())
        
        total_gradients = torch.zeros_like(input_tensor)

        print(f"Generating SmoothGrad (Averaging {n_samples} samples)...")
        for i in range(n_samples):
            # Add random noise
            noise = torch.randn_like(input_tensor) * stdev
            noisy_input = input_tensor + noise
            noisy_input.requires_grad = True
            
            output, _ = self.model(noisy_input)
            score, _ = torch.max(output, dim=1)
            self.model.zero_grad()
            score.backward()
            
            total_gradients += noisy_input.grad.data.abs()

        avg_gradients = total_gradients[0].cpu().mean(dim=0).numpy()
        
        plt.clf()
        plt.imshow(avg_gradients, cmap='hot')
        plt.axis('off')
        plt.title("SmoothGrad (De-Noised Saliency)")

        smoothgrad_pil_img = self._convert_matplot_to_img(plt)
        return smoothgrad_pil_img
    
    def generate_eigen_cam(self, image_path: str | Path) -> Image.Image:
        """
        Generates a Eigen-CAM heatmap. Uses main patterns in the model's feature map to show most important parts.
        """

        img = self._convert_rgb_and_resize(image_path, return_image=True)
        rgb_img = self._convert_rgb_and_resize(image_path, use_scaling=True)

        input_tensor = self._get_input_tensor(img)

        with EigenCAM(model=ClassifierWrapper(self.model).to(self.device), 
                      target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
        return Image.fromarray(cam_image)
        

    def generate_score_cam(self, image_path: str | Path) -> Image.Image:
        """
        Generates a Score-CAM heatmap. This is often more accurate when the model's gradients are noisy or biased.
        """
        
        img = self._convert_rgb_and_resize(image_path, return_image=True)
        rgb_img = self._convert_rgb_and_resize(image_path, use_scaling=True)

        input_tensor = self._get_input_tensor(img)

        # Score-CAM takes a few seconds because it runs a forward pass for every activation map
        with ScoreCAM(model=ClassifierWrapper(self.model).to(self.device), 
                      target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
        return Image.fromarray(cam_image)

    def generate_feature_ablation(self, image_path: str | Path, target_idx: int = None) -> Image.Image:
        """
        Generates a Feature Ablation map.
        Systematically removes superpixels to see which ones are dealbreakers.
        """

        img = self._convert_rgb_and_resize(image_path, get_resized=True)
        img_np = self._convert_rgb_and_resize(image_path, use_scaling=True)
        input_tensor = self._get_input_tensor(img)

        # 2. Create segments (superpixels) to ablate
        # We group pixels so the ablation isn't too slow
        segments = slic(np.array(img), n_segments=50, compactness=10)
        feature_mask = torch.from_numpy(segments).unsqueeze(0).unsqueeze(0).to(self.device)

        if target_idx is None:
            with torch.no_grad():
                logits, _ = self.model(input_tensor)
                target_idx = torch.argmax(logits, dim=1).item()

        # 3. Run Ablation
        ablator = FeatureAblation(ClassifierWrapper(self.model).to(self.device))
        print("Generating Feature Ablation (Systematic Removal)...")
        attributions = ablator.attribute(input_tensor, target=target_idx, feature_mask=feature_mask)

        # 4. Visualize
        attributions_np = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        fig, axis = viz.visualize_image_attr(
            attributions_np,
            img_np,
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            title=f"Feature Ablation (Class: {self.class_names[target_idx]})",
            use_pyplot=False
        )

        fa_pil_img = self._convert_matplot_to_img(fig)
        return fa_pil_img


    def generate_textual_explanation(self, label: str, conf: float, quality_scores: Dict[str, float],
                                      lime_weights: List[Tuple[int, float]], counterfactual_insight: str) -> str:
        """
        Textual XAI. Generates a human-readable narrative explaining the model's decision.
        """
        narrative = [
            f"Primary Classification: {label.upper()}",
            f"Model Confidence: {conf*100:.1f}%",
            ""
        ]

        # Model-Generated Statistical Ranking (LIME), sort the weights to show the top 3 regions the model was most sensitive to
        narrative.append("Model Sensitivity Ranking: (Refer to LIME Plot numbers)")
        narrative.append("The model's decision was mathematically driven by these regions:")
        # Sort weights by absolute importance
        top_features = sorted(lime_weights, key=lambda x: abs(x[1]), reverse=True)[:3]
        for i, (feat_id, weight) in enumerate(top_features):
            impact = "Positive Contribution" if weight > 0 else "Negative"
            narrative.append(f" {i+1}. Region #{feat_id:02} | Impact Score: {abs(weight):.4f} ({impact})")

        # Anchor Logic (From Counterfactual)
        narrative.append("")
        narrative.append("Decision Anchor: (Refer to Counterfactual Plot)")
        # Extract the % change from counterfactual insight string
        improvement_line = counterfactual_insight.splitlines()[-1]
        narrative.append(f"- {improvement_line}")

        # Quality Logic 
        narrative.append("")
        narrative.append("Business Threshold:")
        if label.lower() == "rotten":
            narrative.append("- Critical: AI detected active decay. Immediate Grade C assignment.")
        else:
            narrative.append(f"- Colour: {quality_scores['colour']}% | Size: {quality_scores['size']}% | Ripeness: {quality_scores['ripeness']}%")

        return "\n".join(narrative)

    
    def generate_master_audit_report(self, image_path: str | Path) -> Image.Image:
        """
        Report: Tiles all 10 XAI methods into a single professional document.
        """
        print(f"Generating Audit Report for {Path(image_path).name}...")

        # Gather all Data
        label, conf, scores = self._get_model_outputs(image_path)
        lime_img, lime_weights = self.generate_lime_explanations(image_path)
        cf_img, cf_insight = self.generate_counterfactual(image_path)
        
        # Collect all 10 methods
        original_img = self._convert_rgb_and_resize(image_path, get_resized=True)
        gc_dict = self.generate_gradcam_explanations(image_path)
        m1 = gc_dict['classification']
        m2 = self.generate_eigen_cam(image_path)
        m3 = self.generate_score_cam(image_path)
        m5 = self.generate_integrated_gradient(image_path)
        m6 = self.generate_feature_ablation(image_path)
        m7 = self.generate_shap_explanation(image_path)
        m8 = self.generate_occlusion_explanation(image_path)
        m10 = self.generate_smoothgrad(image_path)

        # Build the Narrative Text
        narrative = self.generate_textual_explanation(label, conf, scores, lime_weights, cf_insight)

        # Create the Layout (Grid on left, Text on right)
        plt.clf()
        fig = plt.figure(figsize=(30, 20), facecolor='#F8F9FA')
        # 3 rows, 5 columns. Columns 0-3 for images, Column 4 for text.
        gs = fig.add_gridspec(3, 5) 

        # Map images to the first 4 columns
        plot_imgs = [original_img, m1, m2, m3, lime_img, m5, m6, m7, m8, cf_img, m10]
        titles = ["Original Image", "1. Grad-CAM", "2. Eigen-CAM", 
            "3. Score-CAM", "4. LIME", "5. Int. Gradients", 
            "6. Feature Ablation", "7. SHAP", "8. Occlusion", 
            "9. Counterfactual", "10. SmoothGrad"]

        # Draw the Images in the first 4 columns
        for i in range(len(plot_imgs)):
            row, col = divmod(i, 4) # This ensures we use 4 columns for images
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(plot_imgs[i])
            ax.set_title(titles[i], fontsize=16, fontweight='bold', pad=10)
            ax.axis('off')

        # Draw the Narrative Text in the 5th column (spanning all rows)
        text_ax = fig.add_subplot(gs[:, 4])
        text_ax.axis('off')
        text_ax.text(0, 0.95, narrative, fontsize=16, verticalalignment='top', 
                     family='monospace', linespacing=1.5,
                     bbox=dict(boxstyle='round,pad=1', facecolor='white', edgecolor='#CCCCCC', alpha=1.0))
        
        fig.suptitle(f"Digital Marketplace: AI Summary\nProduct: {Path(image_path).name}", 
                     fontsize=32, fontweight='bold', color='#2C3E50', y=0.98)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        report_pil = self._convert_matplot_to_img(fig, plt)
        return report_pil
        

if __name__ == "__main__":
    MODEL_PATH = "fresh_rotten_efficientnetv2.pth" 
    TEST_IMAGE_PATH = r"FruitAndVegetableDataset\Fruit And Vegetable Diseases Dataset\Orange__Rotten\rotated_by_15_Screen Shot 2018-06-12 at 11.18.34 PM.png"
    
    if Path(MODEL_PATH).exists() and Path(TEST_IMAGE_PATH).exists():
        xai = ProduceXAI(model_path=MODEL_PATH)
        report = xai.generate_master_audit_report(TEST_IMAGE_PATH).save("ROTTEN_orange_EFFNET.jpg")

        print("\n" + "="*30)
        print("AUDIT COMPLETE")
        print("The 10-Method Transparency Report has been saved.")
        print("="*30)
    else:
        print("Please check your file paths.")