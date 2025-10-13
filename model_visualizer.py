import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


class ModelVisualizer:
    """
    Utility class for logging and visualizing model metrics, embeddings, and feature maps
    using TensorBoard and Matplotlib/Seaborn.

    :param log_dir: Directory to save TensorBoard logs, defaults to "runs/visualizer"
    :type log_dir: str, optional
    :param use_tsne: Whether to use t-SNE instead of PCA for feature visualization, defaults to False
    :type use_tsne: bool, optional
    :param embedding_sample: Number of embedding samples to visualize, defaults to 500
    :type embedding_sample: int, optional
    """

    def __init__(self, log_dir="runs/visualizer", use_tsne=False, embedding_sample=500):
        self.writer = SummaryWriter(log_dir)
        self.use_tsne = use_tsne
        self.embedding_sample = embedding_sample
        self.embeddings, self.labels, self.all_embeddings, self.all_labels = [], [], [], []
        self.hook_handle = None
        self._current_labels = None
        self._capture_embeddings = False

    def register_hook(self, model: nn.Module):
        """
        Register a forward hook on the last layer of the model to capture embeddings.
        Works for both custom CNNs and pretrained models like ResNet.
        """
        def hook_fn(module, input, output):
            if not self._capture_embeddings:
                return
                
            out_cpu = output.detach().cpu()
            self.embeddings.append(out_cpu)
            self.all_embeddings.append(out_cpu)
            if self._current_labels is not None:
                lbls = self._current_labels.cpu()
                if lbls.shape[0] == out_cpu.shape[0]:
                    self.labels.append(lbls)
                    self.all_labels.append(lbls)
                else:
                    print(f"[WARN] Label count {lbls.shape[0]} != embedding count {out_cpu.shape[0]} — skipping this batch.")

        target_layer = None

        if hasattr(model, "base_model"):
            last_block = model.base_model.fc
            if isinstance(last_block, nn.Sequential):
                linear_layers = [m for m in last_block.modules() if isinstance(m, nn.Linear)]
                if len(linear_layers) >= 2:
                    target_layer = linear_layers[-2]
                elif len(linear_layers) >= 1:
                    target_layer = linear_layers[0]
                else:
                    target_layer = last_block
            else:
                target_layer = last_block

        else:
            linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
            if len(linear_layers) >= 2:
                target_layer = linear_layers[-2]
            elif len(linear_layers) == 1:
                target_layer = linear_layers[-1]
            else:
                target_layer = list(model.children())[-1]

        self.hook_handle = target_layer.register_forward_hook(hook_fn)
        print(f"[INFO] Hook registered on layer: {target_layer.__class__.__name__}")
        if isinstance(target_layer, nn.Linear):
            print(f"[INFO] Linear layer output dimension: {target_layer.out_features}")

    def remove_hook(self):
        """
        Remove the registered forward hook to stop capturing embeddings.
        """
        if self.hook_handle:
            self.hook_handle.remove()

    def log_metrics(self, epoch, train_loss, val_loss=None, train_acc=None, val_acc=None):
        """
        Log training and validation metrics to TensorBoard.

        :param epoch: Current epoch
        :type epoch: int
        :param train_loss: Training loss
        :type train_loss: float
        :param val_loss: Validation loss, optional
        :type val_loss: float, optional
        :param train_acc: Training accuracy, optional
        :type train_acc: float, optional
        :param val_acc: Validation accuracy, optional
        :type val_acc: float, optional
        """
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        if train_acc is not None:
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
        if val_acc is not None:
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    def log_embeddings(self, epoch, class_names=None):
        """
        Log captured embeddings to TensorBoard.

        :param epoch: Current epoch
        :type epoch: int
        :param class_names: Optional list of class names for better visualization
        :type class_names: list[str], optional
        """
        if not self.embeddings or not self.labels:
            print(f"[INFO] No embeddings to log for epoch {epoch}")
            return
        
        features = torch.cat(self.embeddings)
        labels = torch.cat(self.labels)
        
        if features.shape[0] != labels.shape[0]:
            print(f"[WARN] Skipping embedding log - shape mismatch: {features.shape[0]} features vs {labels.shape[0]} labels")
            self.embeddings.clear()
            self.labels.clear()
            return
        
        if class_names is not None:
            metadata = [class_names[int(label)] for label in labels]
        else:
            metadata = labels.tolist()
        
        print(f"[INFO] Logging {features.shape[0]} embeddings with dimension {features.shape[1]} for epoch {epoch}")
        
        self.writer.add_embedding(
            features, 
            metadata=metadata,
            global_step=epoch, 
            tag="embeddings"
        )
        
        self.embeddings.clear()
        self.labels.clear()

    def plot_loss_curves(self, train_losses, val_losses, save_path, model_name="Model"):
        """
        Plot training and validation loss curves.

        :param train_losses: List of training losses per epoch
        :type train_losses: list[float]
        :param val_losses: List of validation losses per epoch
        :type val_losses: list[float]
        :param save_path: path to save the plot
        :type save_path: str
        :param model_name: Name of the model, defaults to "Model"
        :type model_name: str, optional
        """
        plt.figure(figsize=(7, 4))
        plt.plot(train_losses, label="Train Loss", linewidth=2.2)
        plt.plot(val_losses, label="Val Loss", linewidth=2.2)
        plt.title(f"Training vs Validation Loss ({model_name})", fontsize=13)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Loss curves saved to {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path, model_name="Model"):
        """
        Plot a confusion matrix.

        :param y_true: Ground truth labels
        :type y_true: array-like
        :param y_pred: Predicted labels
        :type y_pred: array-like
        :param class_names: Name of the classes
        :type class_names: list[str], optional
        :param save_path: path to save the plot
        :type save_path: str
        :param model_name: Name of the model, defaults to "Model"
        :type model_name: str, optional
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def plot_feature_space(self, features, labels, save_path, model_name="Model"):
        """
        Plot feature embeddings in 2D using PCA or t-SNE.

        :param features: Feature tensor from the model
        :type features: torch.Tensor
        :param labels: Corresponding labels for features
        :type labels: array-like
        :param save_path: path to save the plot
        :type save_path: str
        :param model_name: Name of the model, defaults to "Model"
        :type model_name: str, optional
        """
        features = features.detach().cpu().numpy()
        labels = np.array(labels)

        if features.shape[0] > self.embedding_sample:
            idx = np.random.choice(features.shape[0], self.embedding_sample, replace=False)
            features, labels = features[idx], labels[idx]

        if self.use_tsne:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(features)

        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            x=reduced[:, 0], y=reduced[:, 1],
            hue=labels, palette="tab10", s=50, alpha=0.8, edgecolor="none"
        )
        plt.title(f"{model_name} Feature Space ({'t-SNE' if self.use_tsne else 'PCA'})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Feature Embeddings saved to {save_path}")

    def visualize_feature_maps(self, model, input_image, layer_name, save_path, model_name="Model"):
        """
        Visualize the feature maps from a specific convolutional layer.

        :param model: PyTorch model
        :type model: nn.Module
        :param input_image: Input image tensor
        :type input_image: torch.Tensor
        :param layer_name: Name of the layer to visualize
        :type layer_name: str
        :param save_path: path to save the plot
        :type save_path: str
        :param model_name: Name of the model, defaults to "Model"
        :type model_name: str, optional
        """
        activation = {}
        handle = None

        def hook_fn(module, inp, out):
            activation[layer_name] = out.detach().cpu()

        layer_found = False
        available_layers = []
        
        for name, layer in model.named_modules():
            if name:
                available_layers.append(name)
            if name == layer_name:
                handle = layer.register_forward_hook(hook_fn)
                layer_found = True
                print(f"[INFO] Visualizing feature maps from layer: {name}")
                break
        
        if not layer_found:
            print(f"[ERROR] Layer '{layer_name}' not found in model.")
            print(f"[INFO] Available convolutional layers:")
            for name in available_layers:
                if 'conv' in name.lower() or 'layer' in name.lower():
                    print(f"  - {name}")
            return
        
        with torch.no_grad():
            _ = model(input_image)
        
        if handle is not None:
            handle.remove()

        if layer_name not in activation:
            print(f"[ERROR] No activation captured for layer '{layer_name}'")
            return

        act = activation[layer_name][0]
        num_maps = min(8, act.shape[0])

        fig, axes = plt.subplots(1, num_maps+1, figsize=(15, 4))
        
        img_display = input_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        
        axes[0].imshow(img_display)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        
        for i in range(num_maps):
            axes[i+1].imshow(act[i].numpy(), cmap='viridis')
            axes[i+1].axis('off')
            axes[i+1].set_title(f'Map {i+1}')
        
        plt.suptitle(f"Feature Maps from Layer (Model: {model_name}): {layer_name}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Feature maps saved to {save_path}")

    def close(self):
        """
        Close the TensorBoard writer.
        """
        self.writer.close()