import torch
import torch.nn.functional as F
import numpy as np
import cv2
import shap
import matplotlib.pyplot as plt

# --- SHAP for Tabular Data ---
def get_shap_explanations(model, X_train, X_instance):
    """Generates SHAP values for a single prediction."""
    # Using KernelExplainer for compatibility across all 8 algorithms
    # We use a small background sample (50) to keep the UI fast
    explainer = shap.KernelExplainer(model.predict_proba, X_train[:50])
    shap_values = explainer.shap_values(X_instance)
    return shap_values, explainer.expected_value

# --- Grad-CAM for Deep Learning ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture the feature maps and gradients
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        output[:, class_idx].backward()

        # Weight the channels by the gradients
        weights = torch.mean(self.gradients, dim=[2, 3])
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and Normalization
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        return heatmap

def overlay_heatmap(img, heatmap):
    """Overlays the heatmap on the original image for the UI."""
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend the original image with the heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img