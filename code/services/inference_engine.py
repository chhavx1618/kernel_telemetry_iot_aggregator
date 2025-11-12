"""Inference engine for running predictions with machine learning models."""
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

from ..models import InferenceResult, Database, ModelMetadata
from ..config import Config

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Handles model loading and inference operations."""
    
    def __init__(self, db: Database, registry: 'ModelRegistry'):
        """Initialize the inference engine.
        
        Args:
            db: Database instance for logging inferences
            registry: Model registry instance for model management
        """
        self.db = db
        self.registry = registry
        self.runner = None
        self.current_version = None
        self.model_info = None
        self.lock = threading.Lock()
        
        # Load the active model if available
        self.load_active_model()
    
    def load_active_model(self) -> bool:
        """Load the currently active model.
        
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        active_model = self.db.get_active_model_metadata()
        if not active_model:
            logger.warning("No active model found in database")
            return False
            
        model_path = self.registry.get_model_path(active_model.version)
        if not model_path:
            logger.error(f"Active model file not found for version {active_model.version}")
            return False
            
        return self.load_model(active_model.version, model_path)
    
    def load_model(self, version: int, model_path: Path) -> bool:
        """Load a specific model version.
        
        Args:
            version: Model version number
            model_path: Path to the model file
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        with self.lock:
            # Unload current model if any
            if self.runner is not None:
                self.runner.__exit__(None, None, None)
                self.runner = None
                self.current_version = None
                self.model_info = None
            
            try:
                # Initialize the model runner
                logger.info(f"Loading model version {version} from {model_path}")
                self.runner = ImageImpulseRunner(str(model_path))
                
                # Initialize the model (this may take some time)
                self.model_info = self.runner.init()
                logger.info(f"Model loaded: {self.model_info}")
                
                self.current_version = version
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model version {version}: {e}")
                if self.runner:
                    self.runner.__exit__(None, None, None)
                    self.runner = None
                return False
    
    def predict(self, image_path: Path, confidence_threshold: float = Config.CONFIDENCE_THRESHOLD) -> Optional[InferenceResult]:
        """Run inference on an image.
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            InferenceResult containing the prediction results, or None if inference failed
        """
        if not self.runner or not self.current_version:
            logger.error("No model loaded for inference")
            return None
            
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            # Read and preprocess the image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Failed to read image")
                
            # Convert BGR to RGB if needed (depends on model requirements)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run inference
            start_time = time.time()
            features, cropped = self.runner.get_features_from_image(img)
            result = self.runner.classify(features)
            inference_time = time.time() - start_time
            
            # Process results
            detections = []
            confidence_scores = []
            
            # The exact structure depends on your model's output format
            # This is a simplified example - adjust according to your model
            if 'result' in result and 'classification' in result['result']:
                for label, prob in result['result']['classification'].items():
                    if prob >= confidence_threshold:
                        detections.append({
                            'label': label,
                            'confidence': float(prob),
                            'bounding_box': None  # Update if your model provides bbox
                        })
                        confidence_scores.append(float(prob))
            
            # Create inference result
            inference_result = InferenceResult(
                model_version=self.current_version,
                image_name=image_path.name,
                timestamp=datetime.now().isoformat(),
                detections=detections,
                inference_time=inference_time,
                confidence_scores=confidence_scores
            )
            
            # Log the inference
            self.db.log_inference(inference_result)
            
            return inference_result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return None
    
    def visualize_detections(self, image_path: Path, result: InferenceResult, 
                           output_path: Optional[Path] = None) -> Optional[Path]:
        """Draw detection results on the image.
        
        Args:
            image_path: Path to the original image
            result: Inference result from predict()
            output_path: Optional path to save the visualization
            
        Returns:
            Path to the output image, or None if visualization failed
        """
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
            
        try:
            # Read the image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Failed to read image")
            
            # Draw detections
            for detection in result.detections:
                # This is a simple example - adjust based on your detection format
                label = detection.get('label', 'object')
                confidence = detection.get('confidence', 0)
                bbox = detection.get('bounding_box')
                
                if bbox and len(bbox) == 4:  # x, y, w, h
                    x, y, w, h = map(int, bbox)
                    # Draw rectangle
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Draw label background
                    label_size = cv2.getTextSize(f"{label} {confidence:.2f}", 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(
                        img, 
                        (x, y - label_size[1] - 5), 
                        (x + label_size[0], y), 
                        (0, 255, 0), 
                        cv2.FILLED
                    )
                    # Draw label text
                    cv2.putText(
                        img, 
                        f"{label} {confidence:.2f}", 
                        (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0), 
                        1
                    )
            
            # Save or return the result
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), img)
                return output_path
                
            # If no output path, create a temporary file
            temp_path = Path("/tmp") / f"detection_{int(time.time())}.jpg"
            cv2.imwrite(str(temp_path), img)
            return temp_path
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            return None
